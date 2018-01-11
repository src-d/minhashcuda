[![Build Status](https://travis-ci.org/src-d/minhashcuda.svg?branch=master)](https://travis-ci.org/src-d/minhashcuda) [![PyPI](https://img.shields.io/pypi/v/libMHCUDA.svg)](https://pypi.python.org/pypi/libMHCUDA) [![10.5281/zenodo.286955](https://zenodo.org/badge/DOI/10.5281/zenodo.286955.svg)](https://doi.org/10.5281/zenodo.286955)

MinHashCuda
===========

This project is the reimplementation of Weighted MinHash calculation from
[ekzhu/datasketch](https://github.com/ekzhu/datasketch) in NVIDIA CUDA and thus
brings 600-1000x speedup over numpy with [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library)
(Titan X 2016 vs 12-core Xeon E5-1650).
It supports running on multiple GPUs to be even faster, e.g., processing 10Mx12M
matrix with sparsity 0.0014 takes 40 minutes using two Titan Xs.
The produced results are bit-to-bit identical to the reference implementation.
Read the [article](http://blog.sourced.tech/post/minhashcuda/).

The input format is 32-bit float [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) matrix.
The code is optimized for low memory consumption and speed.

What is Weighted MinHash
--------------------------
MinHash can be used to compress unweighted set or binary vector, and estimate
unweighted Jaccard similarity.
It is possible to modify MinHash for
[weighted Jaccard](https://en.wikipedia.org/wiki/Jaccard_index#Generalized_Jaccard_similarity_and_distance)
by expanding each item (or dimension) by its weight.
However this approach does not support real number weights, and
doing so can be very expensive if the weights are very large.
[Weighted MinHash](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf)
is created by Sergey Ioffe, and its performance does not depend on the weights - as
long as the universe of all possible items (or dimension for vectors) is known.
This makes it unsuitable for stream processing, when the knowledge of unseen
items cannot be assumed.

Building
--------
```
cmake -DCMAKE_BUILD_TYPE=Release . && make
```
It requires cudart, curand 7.5 and OpenMP 4.0 capable compiler.
If [numpy](http://www.numpy.org/) headers are not found,
specify the includes path with defining `NUMPY_INCLUDES`.
If you do not want to build the Python native module, add `-D DISABLE_PYTHON=y`.
If CUDA is not automatically found, add `-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0`
(change the path to the actual one).

Python users: if you are using Linux x86-64 and CUDA 7.5, then you can
install this easily:
```
pip install libMHCUDA
```
Otherwise, you'll have to install it from source:
```
pip install git+https://github.com/src-d/minhashcuda.git
```

Testing
-------
`test.py` contains the unit tests based on [unittest](https://docs.python.org/3/library/unittest.html).
They require [datasketch](https://github.com/ekzhu/datasketch) and [scipy](https://github.com/scipy/scipy).

Contributions
-------------

...are welcome! See [CONTRIBUTING](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

License
-------

[Apache 2.0](LICENSE.md)

Python example
--------------
```python
import libMHCUDA
import numpy
from scipy.sparse import csr_matrix

# Prepare the rows
numpy.random.seed(1)
data = numpy.random.randint(0, 100, (6400, 130))
mask = numpy.random.randint(0, 5, data.shape)
data *= (mask >= 4)
del mask
m = csr_matrix(data, dtype=numpy.float32)
del data

# We've got 80% sparse matrix 6400 x 130
# Initialize the hasher aka "generator" with 128 hash samples for every row
gen = libMHCUDA.minhash_cuda_init(m.shape[-1], 128, seed=1, verbosity=1)

# Calculate the hashes. Can be executed several times with different number of rows
hashes = libMHCUDA.minhash_cuda_calc(gen, m)

# Free the resources
libMHCUDA.minhash_cuda_fini(gen)
```
The functions can be easily wrapped into a class (not included).

Python API
----------
Import "libMHCUDA".

```python
def minhash_cuda_init(dim, samples, seed=time(), deferred=False, devices=0, verbosity=0)
```
Creates the hasher.

**dim** integer, the number of dimensions in the input. In other words, length of each weight vector.
        Must be less than 2³².

**samples** integer, the number of hash samples. The more the value, the more precise are the estimates,
            but the larger the hash size and the longer to calculate (linear). Must not be prime
            for performance considerations and less than 2¹⁶.

**seed** integer, the random generator seed for reproducible results.

**deferred** boolean, if True, disables the initialization of WMH parameters with
             random numbers. In that case, the user is expected to call
             minhash_cuda_assign_random_vars() afterwards.

**devices** integer, bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device,
            3 means using first and second device. Special value 0 enables all available devices.
            Default value is 0.

**verbosity** integer, 0 means complete silence, 1 means mere progress logging,
              2 means lots of output.
              
**return** integer, pointer to generator struct (opaque).

```python
def minhash_cuda_calc(gen, matrix, row_start=0, row_finish=0xffffffff)
```
Calculates Weighted MinHash-es. May reallocate memory on GPU but does it's best to reuse the buffers.

**gen** integer, pointer to generator struct obtained from init().

**matrix** `scipy.sparse.csr_matrix` instance, the number of columns must match **dim**.
           The number of rows must be less than 2³¹.
           
**row_start** integer, slice start offset (the index of the first row to process).
              Enables efficient zero-copy sparse matrix slicing.
              
**row_finish** integer, slice finish offset (the index of the row after the last
               one to process). The resulting matrix row slice is [row-start:row_finish].

**return** `numpy.ndarray` of shape (number of matrix rows, **samples**, 2) and dtype uint32.

```python
def minhash_cuda_fini(gen)
```
Disposes any resources allocated by init() and subsequent calc()-s. Generator pointer is invalidated.

**gen** integer, pointer to generator struct obtained from init().

C API
-----
Include "minhashcuda.h".

```C
MinhashCudaGenerator* mhcuda_init(
    uint32_t dim, uint16_t samples, uint32_t seed, int deferred,
    uint32_t devices, int verbosity, MHCUDAResult *status)
```
Initializes the Weighted MinHash generator.

**dim** the number of dimensions in the input. In other words, length of each weight vector.

**samples** he number of hash samples. The more the value, the more precise are the estimates,
            but the larger the hash size and the longer to calculate (linear). Must not be prime
            for performance considerations.

**seed** the random generator seed for reproducible results.

**deferred** if set to anything except 0, disables the initialization of WMH parameters with
             random numbers. In that case, the user is expected to call
             mhcuda_assign_random_vars() afterwards.

**devices** bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device,
            3 means using first and second device. Special value 0 enables all available devices.

**verbosity** 0 means complete silence, 1 means mere progress logging, 2 means lots of output.

**status** pointer to the reported return code. May be nullptr. In case of any error, the
           returned result is nullptr and the code is stored into *status (with nullptr check).

**return** pointer to the allocated generator opaque struct.

```C
MHCUDAResult mhcuda_calc(
    const MinhashCudaGenerator *gen, const float *weights,
    const uint32_t *cols, const uint32_t *rows, uint32_t length,
    uint32_t *output)
```
Calculates the Weighted MinHash-es for the specified CSR matrix.

**gen** pointer to the generator opaque struct obtained from mhcuda_init().
**weights** sparse matrix's values.
**cols** sparse matrix's column indices, must be the same size as weights.
**rows** sparse matrix's row indices. The first element is always 0, the last is
         effectively the size of weights and cols.
**length** the number of rows. "rows" argument must have the size (rows + 1) because of
           the leading 0.
**output** resulting hashes array of size rows x samples x 2.

**return** the status code.

```C
MHCUDAResult mhcuda_fini(MinhashCudaGenerator *gen);
```
Frees any resources allocated by mhcuda_init() and mhcuda_calc(), including device buffers.
Generator pointer is invalidated.

**gen** pointer to the generator opaque struct obtained from mhcuda_init().

**return** the status code.

#### README {#ignore_this_doxygen_anchor}
