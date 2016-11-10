#include <cassert>
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

/*
  weights, cols, rows - CSR format
  plan - execution plan, consists of 2 parts: first is offset table and
         the second is the row indices
*/
__global__ void weighted_minhash_cuda(
    const float *__restrict__ rs, const float *__restrict__ ln_cs,
    const float *__restrict__ betas, const float *__restrict__ weights,
    const uint32_t *__restrict__ cols, const uint32_t *__restrict__ rows,
    const int32_t *__restrict__ plan, const int sample_delta,
    const uint32_t device_row_offset, const uint32_t device_wc_offset,
    uint32_t *__restrict__ hashes) {
  const uint32_t thread_index = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t sample_index = threadIdx.x;
  int32_t row_offset = plan[thread_index];
  int32_t row_border = plan[thread_index + 1];
  if (row_offset == row_border) {
    return;
  }
  const uint32_t sample_offset = sample_index * sample_delta;
  const uint32_t samples = blockDim.x * sample_delta;
  extern __shared__ float shmem[];
  float *volatile lnmins = &shmem[(threadIdx.y * blockDim.x + sample_index) * 3 * sample_delta];
  uint2 *volatile dtmins = reinterpret_cast<uint2 *>(lnmins + sample_delta);
  int32_t row = -1;
  for (uint32_t index = 0, border = 0;; index++) {
    if (index >= border) {
      for (uint32_t s = 0; s < sample_delta; s++) {
        lnmins[s] = FLT_MAX;
      }
      if (row >= 0) {
        for (int s = 0; s < sample_delta; s++) {
          auto hash = reinterpret_cast<uint2 *>(hashes +
              ((row - device_row_offset) * samples + s + sample_offset) * 2);
          *hash = dtmins[s];
        }
      }
      if (row_offset >= row_border) {
        break;
      }
      row = plan[row_offset++];
      index = rows[row - device_row_offset];
      border = rows[row - device_row_offset + 1];
    }
    const float w = logf(weights[index - device_wc_offset]);
    const uint32_t d = cols[index - device_wc_offset];
    int64_t ci = static_cast<int64_t>(sample_offset) * d_dim + d;
    #pragma unroll 4
    for (int s = 0; s < sample_delta; s++, ci += d_dim) {
      float r = rs[ci];
      float beta = betas[ci];
      float t = floorf(w / r + beta);
      float ln_y = (t - beta) * r;
      float ln_a = ln_cs[ci] - ln_y - r;
      if (ln_a < lnmins[s]) {
        lnmins[s] = ln_a;
        dtmins[s] = {d, static_cast<uint32_t>(t)};
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

MHCUDAResult setup_weighted_minhash(
    uint32_t dim, const std::vector<int> &devs, int verbosity) {
  FOR_EACH_DEV(
    CUCH(cudaMemcpyToSymbol(d_dim, &dim, sizeof(dim)),
         mhcudaMemoryCopyError);
  );
  return mhcudaSuccess;
}

MHCUDAResult weighted_minhash(
    const udevptrs<float> &rs, const udevptrs<float> &ln_cs,
    const udevptrs<float> &betas, const udevptrs<float> &weights,
    const udevptrs<uint32_t> &cols, const udevptrs<uint32_t> &rows,
    int samples, const std::vector<int> &sample_deltas,
    const udevptrs<int32_t> &plan, const std::vector<uint32_t> &split,
    const uint32_t *original_rows, const std::vector<uint32_t> &grid_sizes,
    const std::vector<int> &devs, int verbosity, udevptrs<uint32_t> *hashes) {
  FOR_EACH_DEVI(
    int sample_delta = sample_deltas[devi];
    int spt = samples / sample_delta;
    assert(MINHASH_BLOCK_SIZE % spt == 0);
    dim3 block(spt, MINHASH_BLOCK_SIZE / spt, 1);
    dim3 grid(1, grid_sizes[devi], 1);
    auto shmem = 3 * 4 * MINHASH_BLOCK_SIZE * sample_delta;
    uint32_t row_offset = (devi > 0)? split[devi - 1] : 0;
    DEBUG("dev #%d: <<<%d, [%d, %d], %d>>>(%u, %u)\n",
          devs[devi], grid.x, block.x, block.y, shmem,
          static_cast<unsigned>(row_offset),
          static_cast<unsigned>(original_rows[row_offset]));
    weighted_minhash_cuda<<<grid, block, shmem>>>(
        rs[devi].get(), ln_cs[devi].get(), betas[devi].get(),
        weights[devi].get(), cols[devi].get(), rows[devi].get(),
        plan[devi].get(), sample_delta, row_offset, original_rows[row_offset],
        (*hashes)[devi].get());
  );
  return mhcudaSuccess;
}

}  // extern "C"