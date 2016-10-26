#include <cassert>
#include <cinttypes>
#include <algorithm>
#include <map>
#include "private.h"

#include <curand.h>

extern "C" {

struct MinhashCudaGenerator_ {
  MinhashCudaGenerator_(uint32_t dim_, uint16_t samples_,
                        const std::vector<int> &devs_, int verbosity_)
      : dim(dim_), samples(samples_), sizes(devs_.size(), 0),
        lengths(devs_.size(), 0), devs(devs_), verbosity(verbosity_) {}

  udevptrs<float> rs;
  udevptrs<float> ln_cs;
  udevptrs<float> betas;
  mutable udevptrs<float> weights;
  mutable udevptrs<uint32_t> cols;
  mutable udevptrs<uint32_t> rows;
  mutable udevptrs<uint32_t> row_blocks;
  mutable udevptrs<uint32_t> hashes;
  uint32_t dim;
  uint16_t samples;
  mutable std::vector<uint32_t> sizes;
  mutable std::vector<uint32_t> lengths;
  std::vector<uint32_t> shmem_sizes;
  std::vector<int> devs;
  int verbosity;
};

static std::vector<int> setup_devices(uint32_t device, int verbosity) {
  std::vector<int> devs;
  if (device == 0) {
    cudaGetDeviceCount(reinterpret_cast<int *>(&device));
    if (device == 0) {
      return std::move(devs);
    }
    device = (1u << device) - 1;
  }
  for (int dev = 0; device; dev++) {
    if (device & 1) {
      devs.push_back(dev);
      if (cudaSetDevice(dev) != cudaSuccess) {
        INFO("failed to validate device %d", dev);
        devs.pop_back();
      }
    }
    device >>= 1;
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

MinhashCudaGenerator *mhcuda_init(
    uint32_t dim, uint16_t samples, uint32_t seed,
    uint32_t devices, int verbosity, MHCUDAResult *status) {
  DEBUG("arguments: %" PRIu32 " %" PRIu16 " %" PRIu32 " %" PRIu32
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
  auto res = mhcuda_init_internal(gen.get(), seed, devs);
  if (res != mhcudaSuccess) {
    if (status) *status = res;
    return nullptr;
  }
  if (verbosity > 1) {
    res = print_memory_stats(devs);
    if (res != mhcudaSuccess) {
      if (status) *status = res;
      return nullptr;
    }
  }
  res = setup_weighted_minhash(dim, verbosity);
  if (res != mhcudaSuccess) {
    if (status) *status = res;
    return nullptr;
  }
  return gen.release();
}

static std::vector<uint32_t> calc_best_split(
    const MinhashCudaGenerator *gen, const uint32_t *rows, uint32_t length) {
  uint32_t ideal_split = rows[length] / gen->devs.size();
  std::vector<std::vector<uint32_t>> variants;
  for (size_t devi = 0; devi < gen->devs.size(); devi++) {
    uint32_t row = std::upper_bound(
        rows, rows + length + 1, ideal_split * (devi + 1)) - rows;
    std::vector<std::vector<uint32_t>> fork;
    if (row <= length) {
      fork.assign(variants.begin(), variants.end());
    }
    for (auto &v : variants) {
      v.push_back(row - 1);
    }
    if (row <= length) {
      for (auto &v : fork) {
        v.push_back(row);
      }
      variants.insert(variants.end(), fork.begin(), fork.end());
    }
  }
  std::vector<uint32_t> *best = nullptr;
  uint32_t min_cost = 0xFFFFFFFFu;
  for (auto &v : variants) {
    uint32_t cost = 0;
    for (size_t i = 0; i < gen->devs.size(); i++) {
      uint32_t row = v[i], prev_row = (i > 0)? v[i - 1] : 0;
      uint32_t diff = rows[row] - rows[prev_row] - gen->sizes[i];
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
    const uint32_t *cols, const uint32_t *rows,
    const std::vector<uint32_t> &split, std::vector<uint32_t> *rsizes,
    std::vector<uint32_t> *tsizes) {
  int verbosity = gen->verbosity;
  auto &devs = gen->devs;
  FOR_EACH_DEVI(
    uint32_t row = split[devi], prev_row = (devi > 0)? split[devi - 1] : 0;
    rsizes->push_back(row - prev_row);
    if (rsizes->back() > gen->lengths[devi]) {
      DEBUG("resizing rows and hashes: %" PRIu32 " -> %" PRIu32,
          gen->lengths[devi], rsizes->back());
      gen->rows[devi].reset();
      gen->hashes[devi].reset();
      {
        uint32_t *ptr;
        CUCH(cudaMalloc(&ptr, rsizes->back() * sizeof(uint32_t)),
             mhcudaMemoryAllocationFailure);
        gen->rows[devi].reset(ptr);
      }
      {
        uint32_t *ptr;
        CUCH(cudaMalloc(&ptr, rsizes->back() * 2 * sizeof(uint32_t)),
             mhcudaMemoryAllocationFailure);
        gen->hashes[devi].reset(ptr);
      }
      gen->lengths[devi] = rsizes->back();
    }
    CUCH(cudaMemcpyAsync(gen->rows[devi].get(), rows + prev_row,
                         rsizes->back() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice), mhcudaMemoryCopyError);
    CUCH(cudaMemsetAsync(gen->hashes[devi].get(), 0xff,
                         rsizes->back() * 2 * sizeof(uint32_t)),
         mhcudaRuntimeError);
    tsizes->push_back(rows[row] - rows[prev_row]);
    if (tsizes->back() > gen->sizes[devi]) {
      DEBUG("resizing weights and cols: %" PRIu32 " -> %" PRIu32,
            gen->sizes[devi], tsizes->back());
      gen->weights[devi].reset();
      gen->cols[devi].reset();
      {
        float *ptr;
        CUCH(cudaMalloc(&ptr, tsizes->back() * sizeof(float)),
             mhcudaMemoryAllocationFailure);
        gen->weights[devi].reset(ptr);
      }
      {
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
  );
  return mhcudaSuccess;
}

static MHCUDAResult calc_strides(
    const MinhashCudaGenerator *gen, std::vector<uint32_t> *strides,
    std::vector<uint32_t> *shmem_sizes) {
  for (size_t devi = 0; devi < gen->devs.size(); devi++) {
    uint32_t length = gen->lengths[devi];
    if (length * sizeof(uint32_t) > gen->shmem_sizes[devi]) {

    }
  }
  return mhcudaSuccess;
}

MHCUDAResult mhcuda_calc(
    const MinhashCudaGenerator *gen, const float *weights,
    const uint32_t *cols, const uint32_t *rows, uint32_t length,
    uint32_t *output) {
  if (!gen || !weights || !cols || !rows || !output || length == 0) {
    return mhcudaInvalidArguments;
  }
  std::vector<uint32_t> rsizes, tsizes, strides, shmem_sizes;
  {
    std::vector<uint32_t> split = calc_best_split(gen, rows, length);
    RETERR(fill_buffers(gen, weights, cols, rows, split, &rsizes, &tsizes));
  }
  calc_strides(gen, &strides, &shmem_sizes);
  RETERR(weighted_minhash(
      gen->rs, gen->ln_cs, gen->betas, gen->weights, gen->cols, gen->rows,
      gen->row_blocks, rsizes, strides, tsizes, gen->samples, shmem_sizes, gen->devs,
      gen->verbosity, reinterpret_cast<udevptrs<uint64_t> *>(&gen->hashes)));
  return mhcudaSuccess;
}

MHCUDAResult mhcuda_fini(MinhashCudaGenerator *gen) {
  if (gen) {
    delete gen;
  }
  return mhcudaSuccess;
}

}  // extern "C"
