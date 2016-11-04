#ifndef MHCUDA_MINHASHCUDA_H
#define MHCUDA_MINHASHCUDA_H

#include <stdint.h>

extern "C" {

#ifdef __GNUC__
#define MALLOC __attribute__((malloc))
#else
#define MALLOC
#endif

typedef struct MinhashCudaGenerator_ MinhashCudaGenerator;

/// Holds the parameters of the hash generator set in mhcuda_init().
typedef struct {
  uint32_t dim;
  uint16_t samples;
  int verbosity;
} MinhashCudaGeneratorParameters;

/// Enumeration of all possible return codes from API functions.
enum MHCUDAResult {
  mhcudaSuccess,
  mhcudaInvalidArguments,
  mhcudaNoSuchDevice,
  mhcudaMemoryAllocationFailure,
  mhcudaRuntimeError,
  mhcudaMemoryCopyError
};

/*
 * @brief Initializes the Weighted MinHash generator.
 * @param dim The number of dimensions in the input. In other words, length of each weight vector.
 * @param samples The number of hash samples. The more the value, the more precise are the estimates,
 *                but the larger the hash size and the longer to calculate (linear). Must not be prime
 *                for performance considerations.
 * @param seed The random generator seed for reproducible results.
 * @param devices Bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device,
 *                3 means using first and second device. Special value 0 enables all available devices.
 * @param verbosity 0 means complete silence, 1 means mere progress logging, 2 means lots of output.
 * @param status The pointer to the reported return code. May be nullptr. In case of any error, the
 *               returned result is nullptr and the code is stored into *status (with nullptr check).
 * @return The pointer to the allocated generator opaque struct.
 */
MinhashCudaGenerator* mhcuda_init(
    uint32_t dim, uint16_t samples, uint32_t seed,
    uint32_t devices, int verbosity, MHCUDAResult *status) MALLOC;

/*
 * @brief Extracts the parameters for the specified Weighted MinHash generator.
 * @param gen The pointer to the generator opaque struct obtained from mhcuda_init().
 * @return The struct with the parameters (readonly).
 */
MinhashCudaGeneratorParameters mhcuda_get_parameters(
    const MinhashCudaGenerator *gen);

/*
 * @brief Copies the random variable sequences from the device. They are generated in mhcuda_init().
 * @param gen The pointer to the generator opaque struct obtained from mhcuda_init().
 * @param rs "rs" random variable sequence pointer. The shape is dim x samples.
 * @param ln_cs "ln_cs" random variable sequence pointer. The shape is dim x samples.
 * @param betas "betas" random variable sequence pointer. The shape is dim x samples.
 * @return The status code.
 * @note For testing / debugging.
 */
MHCUDAResult mhcuda_retrieve_random_vars(
    const MinhashCudaGenerator *gen, float *rs, float *ln_cs, float *betas);

/*
 * @brief Copies the random variable sequences to the device.
 * @param gen The pointer to the generator opaque struct obtained from mhcuda_init().
 * @param rs "rs" random variable sequence pointer. The shape is dim x samples.
 * @param ln_cs "ln_cs" random variable sequence pointer. The shape is dim x samples.
 * @param betas "betas" random variable sequence pointer. The shape is dim x samples.
 * @return The status code.
 * @note For testing / debugging.
 */
MHCUDAResult mhcuda_assign_random_vars(
    const MinhashCudaGenerator *gen, const float *rs,
    const float *ln_cs, const float *betas);

/*
 * Calculates the Weighted MinHash-es for the specified CSR matrix.
 * @param gen The pointer to the generator opaque struct obtained from mhcuda_init().
 * @param weights Sparse matrix's values.
 * @param cols Sparse matrix's column indices, must be the same size as weights.
 * @param rows Sparse matrix's row indices. The first element is always 0, the last is
 *             effectively the size of weights and cols.
 * @param length The number of rows. "rows" argument must have the size (rows + 1) because of
 *               the leading 0.
 * @param output Resulting hashes array of size rows x samples x 2.
 * @return The status code.
 */
MHCUDAResult mhcuda_calc(
    const MinhashCudaGenerator *gen, const float *weights,
    const uint32_t *cols, const uint32_t *rows, uint32_t length,
    uint32_t *output);

/*
 * Frees any resources allocated by mhcuda_init() and mhcuda_calc(), including
 * device buffers. Generator pointer is invalidated.
 * @param gen The pointer to the generator opaque struct obtained from mhcuda_init().
 * @return The status code.
 */
MHCUDAResult mhcuda_fini(MinhashCudaGenerator *gen);

} // extern "C"

#endif // MHCUDA_MINHASHCUDA_H
