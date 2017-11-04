#include <cassert>
#include <cinttypes>
#include <algorithm>
#include <condition_variable>
#include <map>
#include <thread>
#include "private.h"

#include <curand.h>

extern "C" {

struct MinhashCudaGenerator_ {
  MinhashCudaGenerator_(uint32_t dim_, uint16_t samples_,
                        const std::vector<int> &devs_, int verbosity_)
      : dim(dim_), samples(samples_), weights(devs_.size()),
        cols(devs_.size()), rows(devs_.size()), plans(devs_.size()),
        hashes(devs_.size()), sizes(devs_.size(), 0),
        lengths(devs_.size(), 0), plan_sizes(devs_.size(), 0),
        devs(devs_), verbosity(verbosity_) {}

  udevptrs<float> rs;
  udevptrs<float> ln_cs;
  udevptrs<float> betas;
  uint32_t dim;
  uint16_t samples;
  mutable udevptrs<float> weights;
  mutable udevptrs<uint32_t> cols;
  mutable udevptrs<uint32_t> rows;
  mutable udevptrs<int32_t> plans;
  mutable udevptrs<uint32_t> hashes;
  mutable std::vector<uint32_t> sizes;
  mutable std::vector<uint32_t> lengths;
  mutable std::vector<uint32_t> plan_sizes;
  std::vector<uint32_t> shmem_sizes;
  std::vector<int> devs;
  int verbosity;
};

}  // extern "C"


static std::vector<int> setup_devices(uint32_t devices, int verbosity) {
  std::vector<int> devs;
  if (devices == 0) {
    cudaGetDeviceCount(reinterpret_cast<int *>(&devices));
    if (devices == 0) {
      return std::move(devs);
    }
    devices = (1u << devices) - 1;
  }
  for (int dev = 0; devices; dev++) {
    if (devices & 1) {
      devs.push_back(dev);
      if (cudaSetDevice(dev) != cudaSuccess) {
        INFO("failed to validate device %d", dev);
        devs.pop_back();
      }
      cudaDeviceProp props;
      auto err = cudaGetDeviceProperties(&props, dev);
      if (err != cudaSuccess) {
        INFO("failed to cudaGetDeviceProperties(%d): %s\n",
             dev, cudaGetErrorString(err));
        devs.pop_back();
      }
      if (props.major != (CUDA_ARCH / 10) || props.minor != (CUDA_ARCH % 10)) {
        INFO("compute capability mismatch for device %d: wanted %d.%d, have "
             "%d.%d\n>>>> you may want to build kmcuda with -DCUDA_ARCH=%d "
             "(refer to \"Building\" in README.md)\n",
             dev, CUDA_ARCH / 10, CUDA_ARCH % 10, props.major, props.minor,
             props.major * 10 + props.minor);
        devs.pop_back();
      }
    }
    devices >>= 1;
  }
  if (devs.size() > 1) {
    for (int dev1 : devs) {
      for (int dev2 : devs) {
        if (dev1 <= dev2) {
          continue;
        }
        int access = 0;
        cudaDeviceCanAccessPeer(&access, dev1, dev2);
        if (!access) {
          INFO("warning: p2p %d <-> %d is impossible\n", dev1, dev2);
        }
      }
    }
    for (int dev : devs) {
      cudaSetDevice(dev);
      for (int odev : devs) {
        if (dev == odev) {
          continue;
        }
        auto err = cudaDeviceEnablePeerAccess(odev, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
          INFO("p2p is already enabled on gpu #%d\n", dev);
        } else if (err != cudaSuccess) {
          INFO("warning: failed to enable p2p on gpu #%d: %s\n", dev,
               cudaGetErrorString(err));
        }
      }
    }
  }
  return std::move(devs);
}

static MHCUDAResult print_memory_stats(const std::vector<int> &devs) {
  int verbosity = 0;
  FOR_EACH_DEV(
      size_t free_bytes, total_bytes;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return mhcudaRuntimeError;
      }
      printf("GPU #%d memory: used %zu bytes (%.1f%%), free %zu bytes, "
                 "total %zu bytes\n",
             dev, total_bytes - free_bytes,
             (total_bytes - free_bytes) * 100.0 / total_bytes,
             free_bytes, total_bytes);
  );
  return mhcudaSuccess;
}

static const std::map<curandStatus, const char*> CURAND_ERRORS {
  {CURAND_STATUS_SUCCESS, "CURAND_STATUS_SUCCESS"},
  {CURAND_STATUS_VERSION_MISMATCH, "CURAND_STATUS_VERSION_MISMATCH"},
  {CURAND_STATUS_NOT_INITIALIZED, "CURAND_STATUS_NOT_INITIALIZED"},
  {CURAND_STATUS_ALLOCATION_FAILED, "CURAND_STATUS_ALLOCATION_FAILED"},
  {CURAND_STATUS_TYPE_ERROR, "CURAND_STATUS_TYPE_ERROR"},
  {CURAND_STATUS_OUT_OF_RANGE, "CURAND_STATUS_OUT_OF_RANGE"},
  {CURAND_STATUS_LENGTH_NOT_MULTIPLE, "CURAND_STATUS_LENGTH_NOT_MULTIPLE"},
  {CURAND_STATUS_DOUBLE_PRECISION_REQUIRED, "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"},
  {CURAND_STATUS_LAUNCH_FAILURE, "CURAND_STATUS_LAUNCH_FAILURE"},
  {CURAND_STATUS_PREEXISTING_FAILURE, "CURAND_STATUS_PREEXISTING_FAILURE"},
  {CURAND_STATUS_INITIALIZATION_FAILED, "CURAND_STATUS_INITIALIZATION_FAILED"},
  {CURAND_STATUS_ARCH_MISMATCH, "CURAND_STATUS_ARCH_MISMATCH"},
  {CURAND_STATUS_INTERNAL_ERROR, "CURAND_STATUS_INTERNAL_ERROR"}
};

#define CURANDCH(cuda_call, ret, ...) \
do { \
  auto __res = cuda_call; \
  if (__res != CURAND_STATUS_SUCCESS) { \
    DEBUG("%s\n", #cuda_call); \
    INFO("%s:%d -> %s\n", __FILE__, __LINE__, CURAND_ERRORS.find(__res)->second); \
    __VA_ARGS__; \
    return ret; \
  } \
} while (false)

class CurandGenerator : public unique_devptr_parent<curandGenerator_st> {
public:
    explicit CurandGenerator(curandGenerator_t ptr) : unique_devptr_parent<curandGenerator_st>(
        ptr, [](curandGenerator_t p){ curandDestroyGenerator(p); }) {}
};

static MHCUDAResult mhcuda_init_internal(
    MinhashCudaGenerator *gen, uint32_t seed, const std::vector<int>& devs) {
  int verbosity = gen->verbosity;
  size_t const_size = gen->dim * gen->samples;
  CUMALLOC(gen->rs, const_size);
  CUMALLOC(gen->ln_cs, const_size);
  CUMALLOC(gen->betas, const_size);
  CUCH(cudaSetDevice(devs.back()), mhcudaNoSuchDevice);
  curandGenerator_t rndgen_;
  CURANDCH(curandCreateGenerator(&rndgen_, CURAND_RNG_PSEUDO_DEFAULT),
           mhcudaRuntimeError);
  CurandGenerator rndgen(rndgen_);
  CURANDCH(curandSetPseudoRandomGeneratorSeed(rndgen.get(), seed),
           mhcudaRuntimeError);
  CURANDCH(curandGenerateUniform(rndgen.get(), gen->rs.back().get(), const_size),
           mhcudaRuntimeError);
  CURANDCH(curandGenerateUniform(rndgen.get(), gen->ln_cs.back().get(), const_size),
           mhcudaRuntimeError);
  CURANDCH(curandGenerateUniform(rndgen.get(), gen->betas.back().get(), const_size),
           mhcudaRuntimeError);
  CUCH(gamma_(const_size, gen->ln_cs.back().get(), gen->rs.back().get()),
       mhcudaRuntimeError);
  CURANDCH(curandGenerateUniform(rndgen.get(), gen->ln_cs.back().get(), const_size),
           mhcudaRuntimeError);
  CUCH(gamma_(const_size, gen->betas.back().get(), gen->ln_cs.back().get()),
       mhcudaRuntimeError);
  CURANDCH(curandGenerateUniform(rndgen.get(), gen->betas.back().get(), const_size),
           mhcudaRuntimeError);
  CUCH(log_(const_size, gen->ln_cs.back().get()), mhcudaRuntimeError);
  auto devi = devs.size() - 1;
  FOR_OTHER_DEVS(
    CUP2P(&gen->rs, 0, const_size);
    CUP2P(&gen->ln_cs, 0, const_size);
    CUP2P(&gen->betas, 0, const_size);
  );
  FOR_EACH_DEV(
    cudaDeviceProp props;
    CUCH(cudaGetDeviceProperties(&props, dev), mhcudaRuntimeError);
    gen->shmem_sizes.push_back(props.sharedMemPerBlock);
    DEBUG("GPU #%" PRIu32 " has %d bytes of shared memory per block\n",
          dev, gen->shmem_sizes.back());
  );
  return mhcudaSuccess;
}

extern "C" {

MinhashCudaGenerator *mhcuda_init(
    uint32_t dim, uint16_t samples, uint32_t seed,
    uint32_t devices, int verbosity, MHCUDAResult *status) {
  DEBUG("mhcuda_init: %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIu32
        " %d %p\n", dim, samples, seed, devices, verbosity, status);
  if (dim == 0 || samples == 0) {
    if (status) *status = mhcudaInvalidArguments;
    return nullptr;
  }
  auto devs = setup_devices(devices, verbosity);
  if (devs.empty()) {
    if (status) *status = mhcudaNoSuchDevice;
    return nullptr;
  }
  auto gen = std::unique_ptr<MinhashCudaGenerator>(
      new MinhashCudaGenerator(dim, samples, devs, verbosity));
  #define CHECK_SUCCESS(x) do { \
    auto res = x; \
    if (res != mhcudaSuccess) { \
      if (status) *status = res; \
      return nullptr; \
    } \
  } while(false)
  CHECK_SUCCESS(mhcuda_init_internal(gen.get(), seed, devs));
  if (verbosity > 1) {
    CHECK_SUCCESS(print_memory_stats(devs));
  }
  CHECK_SUCCESS(setup_weighted_minhash(dim, devs, verbosity));
  return gen.release();
  #undef CHECK_SUCCESS
}

MinhashCudaGeneratorParameters mhcuda_get_parameters(
    const MinhashCudaGenerator *gen) {
  if (gen == nullptr) {
    return {};
  }
  return MinhashCudaGeneratorParameters {
      .dim=gen->dim, .samples=gen->samples, .verbosity=gen->verbosity
  };
}

MHCUDAResult mhcuda_retrieve_random_vars(
    const MinhashCudaGenerator *gen, float *rs, float *ln_cs, float *betas) {
  if (!gen || !rs || !ln_cs || !betas) {
    return mhcudaInvalidArguments;
  }
  int verbosity = gen->verbosity;
  auto &devs = gen->devs;
  size_t const_size = gen->dim * gen->samples * sizeof(float);
  CUCH(cudaSetDevice(devs[0]), mhcudaNoSuchDevice);
  CUCH(cudaMemcpyAsync(rs, gen->rs[0].get(), const_size, cudaMemcpyDeviceToHost),
       mhcudaMemoryCopyError);
  CUCH(cudaMemcpyAsync(ln_cs, gen->ln_cs[0].get(), const_size, cudaMemcpyDeviceToHost),
       mhcudaMemoryCopyError);
  CUCH(cudaMemcpy(betas, gen->betas[0].get(), const_size, cudaMemcpyDeviceToHost),
       mhcudaMemoryCopyError);
  return mhcudaSuccess;
}

MHCUDAResult mhcuda_assign_random_vars(
    const MinhashCudaGenerator *gen, const float *rs,
    const float *ln_cs, const float *betas) {
  if (!gen || !rs || !ln_cs || !betas) {
    return mhcudaInvalidArguments;
  }
  int verbosity = gen->verbosity;
  auto &devs = gen->devs;
  size_t const_size = gen->dim * gen->samples;
  CUMEMCPY_H2D_ASYNC(gen->rs, 0, rs, const_size);
  CUMEMCPY_H2D_ASYNC(gen->ln_cs, 0, ln_cs, const_size);
  CUMEMCPY_H2D_ASYNC(gen->betas, 0, betas, const_size);
  return mhcudaSuccess;
}

}  // extern "C"

static std::vector<uint32_t> calc_best_split(
    const uint32_t *rows, uint32_t length, const std::vector<int> &devs,
    const std::vector<uint32_t> &sizes) {
  uint32_t ideal_split = rows[length] / devs.size();
  std::vector<std::vector<uint32_t>> variants;
  for (size_t devi = 0; devi < devs.size(); devi++) {
    uint32_t row = std::upper_bound(
        rows, rows + length + 1, ideal_split * (devi + 1)) - rows;
    std::vector<std::vector<uint32_t>> fork;
    if (row <= length) {
      fork.assign(variants.begin(), variants.end());
    }
    if (!variants.empty()) {
      for (auto &v : variants) {
        v.push_back(row - 1);
      }
    } else {
      variants.push_back({row - 1});
    }
    if (row <= length) {
      if (!fork.empty()) {
        for (auto &v : fork) {
          v.push_back(row);
        }
      } else {
        fork.push_back({row});
      }
      variants.insert(variants.end(), fork.begin(), fork.end());
    }
  }
  assert(!variants.empty());
  std::vector<uint32_t> *best = nullptr;
  uint32_t min_cost = 0xFFFFFFFFu;
  for (auto &v : variants) {
    uint32_t cost = 0;
    for (size_t i = 0; i < devs.size(); i++) {
      uint32_t row = v[i], prev_row = (i > 0)? v[i - 1] : 0;
      uint32_t diff = rows[row] - rows[prev_row] - sizes[i];
      if (diff > 0) {
        cost += diff * diff;
      }
    }
    if (cost < min_cost) {
      best = &v;
      min_cost = cost;
    }
  }
  return *best;
}

static MHCUDAResult fill_buffers(
    const MinhashCudaGenerator *gen, const float *weights,
    const uint32_t *cols, const uint32_t *rows, const std::vector<uint32_t> &split,
    std::vector<uint32_t> *rsizes, std::vector<uint32_t> *tsizes) {
  int verbosity = gen->verbosity;
  auto &devs = gen->devs;
  for (size_t devi = 0; devi < devs.size(); devi++) {
    CUCH(cudaSetDevice(devs[devi]), mhcudaNoSuchDevice);
    uint32_t row = split[devi], prev_row = (devi > 0) ? split[devi - 1] : 0;
    rsizes->push_back(row - prev_row);
    if (rsizes->back() > gen->lengths[devi]) {
      DEBUG("resizing rows and hashes: %" PRIu32 " -> %" PRIu32 "\n",
            gen->lengths[devi], rsizes->back());
      gen->rows[devi].reset();
      gen->hashes[devi].reset();
      {
        gen->rows[devi].reset();
        uint32_t *ptr;
        CUCH(cudaMalloc(&ptr, (rsizes->back() + 1) * sizeof(uint32_t)),
             mhcudaMemoryAllocationFailure);
        gen->rows[devi].reset(ptr);
      }
      {
        gen->hashes[devi].reset();
        uint32_t *ptr;
        CUCH(cudaMalloc(&ptr, rsizes->back() * gen->samples * sizeof(uint64_t)),
             mhcudaMemoryAllocationFailure);
        gen->hashes[devi].reset(ptr);
      }
      gen->lengths[devi] = rsizes->back();
    }
    CUCH(cudaMemcpyAsync(gen->rows[devi].get(), rows + prev_row,
                         (rsizes->back() + 1) * sizeof(uint32_t),
                         cudaMemcpyHostToDevice), mhcudaMemoryCopyError);
#ifndef NDEBUG
    CUCH(cudaMemsetAsync(gen->hashes[devi].get(), 0xff,
                         rsizes->back() * gen->samples * 2 * sizeof(uint32_t)),
         mhcudaRuntimeError);
#endif
    tsizes->push_back(rows[row] - rows[prev_row]);
    if (tsizes->back() > gen->sizes[devi]) {
      DEBUG("resizing weights and cols: %" PRIu32 " -> %" PRIu32 "\n",
            gen->sizes[devi], tsizes->back());
      gen->weights[devi].reset();
      gen->cols[devi].reset();
      {
        gen->weights[devi].reset();
        float *ptr;
        CUCH(cudaMalloc(&ptr, tsizes->back() * sizeof(float)),
             mhcudaMemoryAllocationFailure);
        gen->weights[devi].reset(ptr);
      }
      {
        gen->cols[devi].reset();
        uint32_t *ptr;
        CUCH(cudaMalloc(&ptr, tsizes->back() * sizeof(uint32_t)),
             mhcudaMemoryAllocationFailure);
        gen->cols[devi].reset(ptr);
      }
      gen->sizes[devi] = tsizes->back();
    }
    CUCH(cudaMemcpyAsync(gen->weights[devi].get(), weights + rows[prev_row],
                         tsizes->back() * sizeof(float),
                         cudaMemcpyHostToDevice), mhcudaMemoryCopyError);
    CUCH(cudaMemcpyAsync(gen->cols[devi].get(), cols + rows[prev_row],
                         tsizes->back() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice), mhcudaMemoryCopyError);
  }
  return mhcudaSuccess;
}

static void binpack(
    const MinhashCudaGenerator *gen, const uint32_t *rows,
    const std::vector<uint32_t> &split, const std::vector<int> &sample_deltas,
    std::vector<std::vector<int32_t>> *plans, std::vector<uint32_t> *grid_sizes) {
  // https://blog.sourced.tech/post/minhashcuda/
  const int32_t ideal_binavgcount = 20;
  auto &devs = gen->devs;
  int verbosity = gen->verbosity;
  plans->resize(devs.size());
  grid_sizes->resize(devs.size());
  #pragma omp parallel for
  for (size_t devi = 0; devi < devs.size(); devi++) {
    uint32_t last_row = split[devi], first_row = (devi > 0) ? split[devi - 1] : 0;
    std::vector<std::tuple<int32_t, uint32_t>> blocks;
    blocks.reserve(last_row - first_row);
    for (uint32_t i = first_row; i < last_row; i++) {
      blocks.emplace_back(rows[i + 1] - rows[i], i);
    }
    std::sort(blocks.rbegin(), blocks.rend()); // reverse order
    int32_t max = std::get<0>(blocks.front());
    uint32_t size = rows[last_row] - rows[first_row];
    int32_t avg = size / blocks.size();
    int32_t blockDim = (MINHASH_BLOCK_SIZE * sample_deltas[devi]) / gen->samples;
    assert(blockDim > 0);
    int32_t bintotal = ceilf(static_cast<float>(size) / blockDim);
    int32_t max_binavgcount = ceilf(static_cast<float>(bintotal) / avg);
    int32_t binavgcount = max_binavgcount;
    for (int i = 2; binavgcount > ideal_binavgcount &&
         i <= max_binavgcount / ideal_binavgcount; i++) {
      binavgcount = max_binavgcount / i;
    }
    int32_t binsize = std::max(binavgcount * avg, max);
    // this is an initial approximation - the real life is of course tougher
    // we are going to get some imbalance though we greedily try to reduce it
    std::vector<std::pair<int32_t, std::vector<uint32_t>>> bins(
        ceilf(static_cast<float>(size) / (binsize * blockDim)) * blockDim);
    assert(bins.size() > 0 && bins.size() % blockDim == 0);
    DEBUG("dev #%d: binsize %d, bins %zu\n", devs[devi], binsize, bins.size());
    grid_sizes->at(devi) = bins.size() / blockDim;
    for (auto &block : blocks) {
      std::pop_heap(bins.begin(), bins.end());
      auto &bin = bins.back();
      bin.first -= std::get<0>(block); // max heap
      bin.second.push_back(std::get<1>(block));
      std::push_heap(bins.begin(), bins.end());
    }
    std::sort_heap(bins.begin(), bins.end());
#ifndef NDEBUG
    if (verbosity > 1) {
      printf("dev #%d imbalance: ", devs[devi]);
      for (uint32_t i = 0; i < bins.size(); i++) {
        if (i % blockDim == 0 && i > 0) {
          int32_t delta = bins[i].first - bins[i - blockDim].first;
          printf("(%d %d%%) ", delta, -(delta * 100) / bins[i - blockDim].first);
        }
      }
      printf("\n");
    }
#endif
    auto &plan = plans->at(devi);
    plan.resize(bins.size() + 1 + blocks.size());
    uint32_t offset = bins.size() + 1;
    for (uint32_t i = 0; i < bins.size(); i++) {
      plan[i] = offset;
      for (auto row : bins[i].second) {
        plan[offset++] = row;
      }
    }
    plan[bins.size()] = offset;  // end offset equals to the previous
  }
}

static MHCUDAResult fill_plans(
    const MinhashCudaGenerator *gen, const std::vector<std::vector<int32_t>> &plans) {
  int verbosity = gen->verbosity;
  auto &devs = gen->devs;
  assert(plans.size() == devs.size());
  for (size_t devi = 0; devi < devs.size(); devi++) {
    CUCH(cudaSetDevice(devs[devi]), mhcudaNoSuchDevice);
    auto plan_size = plans[devi].size();
    if (gen->plan_sizes[devi] < plan_size) {
      gen->plans[devi].reset();
      int32_t *ptr;
      CUCH(cudaMalloc(&ptr, plan_size * sizeof(int32_t)), mhcudaMemoryAllocationFailure);
      gen->plans[devi].reset(ptr);
      gen->plan_sizes[devi] = plan_size;
    }
    CUCH(cudaMemcpyAsync(gen->plans[devi].get(), plans[devi].data(),
                         plan_size * sizeof(int32_t),
                         cudaMemcpyHostToDevice), mhcudaMemoryCopyError);
  }
  return mhcudaSuccess;
}


static void dump_vector(const std::vector<uint32_t> &vec, const char *name) {
  printf("%s: ", name);
  for (size_t i = 0; i < vec.size() - 1; i++) {
    printf("%" PRIu32 ", ", vec[i]);
  }
  printf("%" PRIu32 "\n", vec.back());
}

static void dump_vectors(const std::vector<std::vector<int32_t>> &vec,
                         const char *name) {
  printf("%s:\n", name);
  for (size_t vi = 0; vi < vec.size(); vi++) {
    printf("[%zu] ", vi);
    auto &subvec = vec[vi];
    auto last = std::min(subvec.size() - 1, static_cast<size_t>(9));
    for (size_t i = 0; i < last; i++) {
      printf("%" PRIi32 ", ", subvec[i]);
    }
    printf("%" PRIi32, subvec[last]);
    if (last < subvec.size() - 1) {
      printf("...\n");
    } else {
      printf("\n");
    }
  }
}

extern "C" {

MHCUDAResult mhcuda_calc(
    const MinhashCudaGenerator *gen, const float *weights,
    const uint32_t *cols, const uint32_t *rows, uint32_t length,
    uint32_t *output) {
  if (!gen || !weights || !cols || !rows || !output || length == 0) {
    return mhcudaInvalidArguments;
  }
  int verbosity = gen->verbosity;
  DEBUG("mhcuda_calc: %p %p %p %p %" PRIu32 " %p\n", gen, weights, cols,
        rows, length, output);
  auto &devs = gen->devs;
  INFO("Preparing...\n");
  auto split = calc_best_split(rows, length, gen->devs, gen->sizes);
  if (verbosity > 1) {
    dump_vector(split, "split");
  }
  std::vector<uint32_t> rsizes, tsizes, grid_sizes;
  std::vector<std::vector<int32_t>> plans;
  RETERR(fill_buffers(gen, weights, cols, rows, split, &rsizes, &tsizes));
  std::vector<int> sample_deltas;
  int samples = gen->samples;
  for (auto shmem_size : gen->shmem_sizes) {
    int sdmax = shmem_size / (3 * 4 * MINHASH_BLOCK_SIZE);
    assert(sdmax > 0);
    int sd = sdmax + 1;
    for (int i = 1; i <= samples && sd > sdmax; i++) {
      if (samples % i == 0) {
        int try_sd = samples / i;
        if (try_sd % 2 == 0) {
          sd = try_sd;
        }
      }
    }
    if (sd > sdmax) {
      return mhcudaInvalidArguments;
    }
    sample_deltas.push_back(sd);
  }
  binpack(gen, rows, split, sample_deltas, &plans, &grid_sizes);
  if (verbosity > 1) {
    dump_vectors(plans, "plans");
    dump_vector(grid_sizes, "grid_sizes");
  }
  RETERR(fill_plans(gen, plans));
  INFO("Executing the CUDA kernel...\n");
  RETERR(weighted_minhash(
      gen->rs, gen->ln_cs, gen->betas, gen->weights, gen->cols, gen->rows,
      samples, sample_deltas, gen->plans, split, rows, grid_sizes, devs,
      verbosity, &gen->hashes));
  FOR_EACH_DEVI(
    auto size = rsizes[devi] * gen->samples * 2;
    CUCH(cudaMemcpyAsync(output, gen->hashes[devi].get(),
                         size * sizeof(uint32_t), cudaMemcpyDeviceToHost),
         mhcudaMemoryCopyError);
    output += size;
  );
  SYNC_ALL_DEVS;
  INFO("mhcuda - success\n");
  return mhcudaSuccess;
}

MHCUDAResult mhcuda_fini(MinhashCudaGenerator *gen) {
  if (gen) {
    delete gen;
  }
  return mhcudaSuccess;
}

}  // extern "C"
