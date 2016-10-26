#ifndef MHCUDA_PRIVATE_H
#define MHCUDA_PRIVATE_H

#include "minhashcuda.h"
#include <cmath>
#include <tuple>
#include "wrappers.h"

#define INFO(...) do { if (verbosity > 0) { printf(__VA_ARGS__); } } while (false)
#define DEBUG(...) do { if (verbosity > 1) { printf(__VA_ARGS__); } } while (false)

#define CUERRSTR() cudaGetErrorString(cudaGetLastError())

#define CUCH(cuda_call, ret, ...) \
do { \
  auto __res = cuda_call; \
  if (__res != 0) { \
    DEBUG("%s\n", #cuda_call); \
    INFO("%s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(__res)); \
    __VA_ARGS__; \
    return ret; \
  } \
} while (false)

#define RETERR(call, ...) \
do { \
  auto __res = call; \
  if (__res != 0) { \
    __VA_ARGS__; \
    return __res; \
  } \
} while (false)

#define FOR_EACH_DEV(...) do { for (int dev : devs) { \
  CUCH(cudaSetDevice(dev), mhcudaNoSuchDevice); \
  __VA_ARGS__; \
} } while(false)

#define FOR_EACH_DEVI(...) do { for (size_t devi = 0; devi < devs.size(); devi++) { \
  CUCH(cudaSetDevice(devs[devi]), mhcudaNoSuchDevice); \
  __VA_ARGS__; \
} } while(false)

#define SYNC_ALL_DEVS do { \
if (devs.size() > 1) { \
FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), mhcudaRuntimeError)); \
} } while (false)

#define CUMEMCPY_D2H_ASYNC(dst, dst_stride, src, src_offset, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemcpyAsync( \
      dst + dst_stride * devi, (src)[devi].get() + src_offset, \
      (size) * sizeof(std::remove_reference<decltype(src)>::type::value_type \
          ::element_type), \
      cudaMemcpyDeviceToHost), \
                     mhcudaMemoryCopyError)); \
} while(false)

#define CUMEMCPY_D2H(dst, src, size) do { \
  CUMEMCPY_D2H_ASYNC(dst, src, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), mhcudaMemoryCopyError)); \
} while(false)

#define CUMEMCPY_H2D_ASYNC(dst, dst_offset, src, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemcpyAsync( \
      (dst)[devi].get() + dst_offset, src, \
      (size) * sizeof(std::remove_reference<decltype(dst)>::type::value_type \
          ::element_type), \
      cudaMemcpyHostToDevice), \
                     mhcudaMemoryCopyError)); \
} while(false)

#define CUMEMCPY_H2D(dst, src, size) do { \
  CUMEMCPY_H2D_ASYNC(dst, src, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), mhcudaMemoryCopyError)); \
} while(false)

#define CUMEMCPY_D2D_ASYNC(dst, dst_offset, src, src_offset, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemcpyAsync( \
      (dst)[devi].get() + dst_offset, (src)[devi].get() + src_offset, \
      (size) * sizeof(std::remove_reference<decltype(dst)>::type::value_type \
          ::element_type), \
      cudaMemcpyDeviceToDevice), \
                     mhcudaMemoryCopyError)); \
} while(false)

#define CUMEMCPY_D2D(dst, dst_offset, src, src_offset, size) do { \
  CUMEMCPY_D2D_ASYNC(dst, dst_offset, src, src_offset, size); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), mhcudaMemoryCopyError)); \
} while(false)

#define CUMALLOC_ONEN(dest, size, name) do { \
  void *__ptr; \
  CUCH(cudaMalloc( \
      &__ptr, \
      (size) * sizeof(std::remove_reference<decltype(dest)>::type::value_type \
          ::element_type)), \
       mhcudaMemoryAllocationFailure, \
       INFO("failed to allocate %zu bytes for " name "\n", \
            static_cast<size_t>(size))); \
  (dest).emplace_back(reinterpret_cast<std::remove_reference<decltype(dest)> \
      ::type::value_type::element_type *>(__ptr)); \
} while(false)

#define CUMALLOC_ONE(dest, size) CUMALLOC_ONEN(dest, size, #dest)

#define CUMALLOCN(dest, size, name) do { \
  FOR_EACH_DEV(CUMALLOC_ONEN(dest, size, name)); \
} while(false)

#define CUMALLOC(dest, size) CUMALLOCN(dest, size, #dest)

#define CUMEMSET(dst, val, size) do { \
  FOR_EACH_DEVI(CUCH(cudaMemsetAsync( \
      (dst)[devi].get(), val, \
      size * sizeof(std::remove_reference<decltype(dst)>::type::value_type::element_type)), \
                     mhcudaRuntimeError)); \
  FOR_EACH_DEV(CUCH(cudaDeviceSynchronize(), mhcudaRuntimeError)); \
} while(false)

#define FOR_OTHER_DEVS(...) do { \
  for (size_t odevi = 0; odevi < devs.size(); odevi++) { \
    if (odevi == devi) { \
      continue; \
    } \
    __VA_ARGS__; \
  } } while(false)

#define CUP2P(what, offset, size) do { \
  CUCH(cudaMemcpyPeerAsync( \
      (*what)[odevi].get() + offset, devs[odevi], (*what)[devi].get() + offset, \
      devs[devi], (size) * sizeof(std::remove_reference<decltype(*what)>::type \
      ::value_type::element_type)), \
       mhcudaMemoryCopyError); \
} while(false)

extern "C" {
  cudaError_t gamma_(uint32_t size, const float *v1, float *v2);
  cudaError_t log_(uint32_t size, float *v);
  MHCUDAResult setup_weighted_minhash(uint32_t dim, int verbosity);
  MHCUDAResult weighted_minhash(
    const udevptrs<float> &rs, const udevptrs<float> &ln_cs,
    const udevptrs<float> &betas, const udevptrs<float> &weights,
    const udevptrs<uint32_t> &cols, const udevptrs<uint32_t> &rows,
    const udevptrs<uint32_t> &row_blocks, const std::vector<uint32_t> &rsizes,
    const std::vector<uint32_t> &strides, const std::vector<uint32_t> &tsizes,
    int samples, const std::vector<uint32_t> &shmem_sizes,
    const std::vector<int> &devs, int verbosity, udevptrs<uint64_t> *hashes);
}

#define MINHASH_BLOCK_SIZE 512

#endif  // MHCUDA_PRIVATE_H
