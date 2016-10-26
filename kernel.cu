#include <cfloat>
#include "private.h"

__constant__ uint32_t d_dim;

__global__ void gamma_cuda(uint32_t size, const float *__restrict__ v1, float *v2) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  v2[index] = -logf(v1[index] * v2[index]);
}

__global__ void log_cuda(uint32_t size, float *v) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  v[index] = logf(v[index]);
}

// uint64_t is a typedef and not understood by atomicMin intrinsic
using uint64_cu = unsigned long long int;
static_assert(sizeof(uint64_cu) == 8, "unsigned long long int in CUDA must be 64-bit");

/*
  weights, cols, rows - CSR format
  size - number of matrix rows; rows array contains (size + 1) elements
  row_blocks - index of rows for blocks
*/
__global__ void weighted_minhash_cuda(
    const float *__restrict__ rs, const float *__restrict__ ln_cs,
    const float *__restrict__ betas, const float *__restrict__ weights,
    const uint32_t *__restrict__ cols, const uint32_t *__restrict__ rows,
    const uint32_t *__restrict__ row_blocks, uint32_t size, uint32_t stride,
    int samples, uint64_cu *__restrict__ hashes) {
  uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t srow = (blockIdx.x > 0? row_blocks[blockIdx.x - 1] : 0);
  uint32_t frow = row_blocks[blockIdx.x];
  extern __shared__ uint32_t shared_rows[];
  {
    uint32_t shmem_stride = ceilf((frow - srow) / blockDim.x);
    uint32_t shmem_start = threadIdx.x * shmem_stride;
    uint32_t shmem_finish = min(shmem_start + shmem_stride, frow);
    uint32_t offset = rows[0];
    for (uint32_t i = shmem_start; i < shmem_finish; i++) {
      shared_rows[i] = rows[i] - offset;
    }
  }
  __syncthreads();

  frow -= srow;
  srow = 0;
  uint32_t start_index = thread_index * stride;
  uint32_t finish_index = min(start_index + stride, shared_rows[frow - 1]);
  if (start_index >= finish_index) {
    return;
  }

  while (srow < frow - 1) {
     uint32_t middle = (srow + frow) << 1;
     uint32_t middle_index = shared_rows[middle];
     if (start_index < middle_index) {
       frow = middle;
     } else {
       srow = middle;
     }
  }

  // srow >= rows[srow] - start row index (may be partial)
  //  |
  //  $------------$------$------$---------------$
  //         |                             |
  //    start_index                   finish_index

  for (int s = 0; s < samples; s++) {
    uint32_t row = srow;
    uint64_cu hash = 0xDEADF00DDEADF00Dllu;
    float ln_amin = FLT_MAX;
    for (uint32_t index = start_index; index < finish_index; index++) {
      float w = weights[index];
      float d = cols[index];
      int64_t ci = s; ci *= d_dim; ci += d;
      float r = rs[ci];
      float beta = betas[ci];
      float t = floorf(logf(w) / r + beta);
      float ln_y = (t - beta) * r;
      float ln_a = ln_cs[ci] - ln_y - r;
      if (ln_a < ln_amin) {
        hash = (static_cast<uint64_cu>(d) << 32) | static_cast<uint64_cu>(t);
      }
      uint32_t next_index = index + 1;
      if (next_index == shared_rows[row + 1] || next_index == finish_index) {
        atomicMin(hashes + (row - 1) * samples + s, hash);
        row++;
        ln_amin = FLT_MAX;
      }
    }
  }
}

extern "C" {

cudaError_t gamma_(uint32_t size, const float *v1, float *v2) {
  dim3 block(1024, 1, 1);
  dim3 grid(size / block.x + 1, 1, 1);
  gamma_cuda<<<grid, block>>>(size, v1, v2);
  RETERR(cudaDeviceSynchronize());
  return cudaSuccess;
}

cudaError_t log_(uint32_t size, float *v) {
  dim3 block(1024, 1, 1);
  dim3 grid(size / block.x + 1, 1, 1);
  log_cuda<<<grid, block>>>(size, v);
  RETERR(cudaDeviceSynchronize());
  return cudaSuccess;
}

MHCUDAResult setup_weighted_minhash(uint32_t dim, int verbosity) {
  CUCH(cudaMemcpyToSymbol(d_dim, &dim, sizeof(dim)),
       mhcudaMemoryCopyError);
  return mhcudaSuccess;
}

MHCUDAResult weighted_minhash(
    const udevptrs<float> &rs, const udevptrs<float> &ln_cs,
    const udevptrs<float> &betas, const udevptrs<float> &weights,
    const udevptrs<uint32_t> &cols, const udevptrs<uint32_t> &rows,
    const udevptrs<uint32_t> &row_blocks, const std::vector<uint32_t> &rsizes,
    const std::vector<uint32_t> &strides, const std::vector<uint32_t> &tsizes,
    int samples, const std::vector<uint32_t> &shmem_sizes,
    const std::vector<int> &devs, int verbosity, udevptrs<uint64_t> *hashes) {
  FOR_EACH_DEVI(
    dim3 block(MINHASH_BLOCK_SIZE, 1, 1);
    dim3 grid(ceilf(static_cast<float>(tsizes[devi]) / block.x), 1, 1);
    weighted_minhash_cuda<<<grid, block, shmem_sizes[devi]>>>(
        rs[devi].get(), ln_cs[devi].get(), betas[devi].get(),
        weights[devi].get(), cols[devi].get(), rows[devi].get(),
        row_blocks[devi].get(), rsizes[devi], strides[devi], samples,
        reinterpret_cast<uint64_cu *>((*hashes)[devi].get()));
  );
  return mhcudaSuccess;
}

}  // extern "C"