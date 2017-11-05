#include <memory>
#include <unordered_map>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "minhashcuda.h"

static char module_docstring[] =
    "This module provides fast Weighted Minhash implementation which uses CUDA.";
static char minhash_cuda_init_docstring[] =
    "Prepares Weighted Minhash internal state on GPU.";
static char minhash_cuda_retrieve_vars_docstring[] =
    "Copies the random variables rs, ln_cs and betas from the generator to the host.";
static char minhash_cuda_assign_vars_docstring[] =
    "Assigns random variables rs, ln_cs and betas to the generator. "
    "Used for testing purposes since those variables are already set "
    "in minhash_cuda_init().";
static char minhash_cuda_calc_docstring[] =
    "Calculates Weighted Minhashes.";
static char minhash_cuda_fini_docstring[] =
    "Disposes Weighted Minhash internal state on GPU.";

static PyObject *py_minhash_cuda_init(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_minhash_cuda_retrieve_vars(PyObject *self, PyObject *args);
static PyObject *py_minhash_cuda_assign_vars(PyObject *self, PyObject *args);
static PyObject *py_minhash_cuda_calc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_minhash_cuda_fini(PyObject *self, PyObject *args);

static PyMethodDef module_functions[] = {
  {"minhash_cuda_init", reinterpret_cast<PyCFunction>(py_minhash_cuda_init),
   METH_VARARGS | METH_KEYWORDS, minhash_cuda_init_docstring},
  {"minhash_cuda_retrieve_vars", reinterpret_cast<PyCFunction>(py_minhash_cuda_retrieve_vars),
   METH_VARARGS, minhash_cuda_retrieve_vars_docstring},
  {"minhash_cuda_assign_vars", reinterpret_cast<PyCFunction>(py_minhash_cuda_assign_vars),
   METH_VARARGS, minhash_cuda_assign_vars_docstring},
  {"minhash_cuda_calc", reinterpret_cast<PyCFunction>(py_minhash_cuda_calc),
   METH_VARARGS | METH_KEYWORDS, minhash_cuda_calc_docstring},
  {"minhash_cuda_fini", reinterpret_cast<PyCFunction>(py_minhash_cuda_fini),
   METH_VARARGS, minhash_cuda_fini_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_libMHCUDA(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "libMHCUDA",         /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  return m;
}
}

template <typename O>
using pyobj_parent = std::unique_ptr<O, std::function<void(O*)>>;

template <typename O>
class _pyobj : public pyobj_parent<O> {
 public:
  _pyobj() : pyobj_parent<O>(
      nullptr, [](O *p){ if (p) Py_DECREF(p); }) {}
  explicit _pyobj(PyObject *ptr) : pyobj_parent<O>(
      reinterpret_cast<O *>(ptr), [](O *p){ if(p) Py_DECREF(p); }) {}
  void reset(PyObject *p) noexcept {
    pyobj_parent<O>::reset(reinterpret_cast<O*>(p));
  }
};

using pyobj = _pyobj<PyObject>;
using pyarray = _pyobj<PyArrayObject>;

static void set_cuda_malloc_error() {
  PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory on GPU");
}

static void set_cuda_device_error() {
  PyErr_SetString(PyExc_ValueError, "No such CUDA device exists");
}

static void set_cuda_memcpy_error() {
  PyErr_SetString(PyExc_RuntimeError, "cudaMemcpy failed");
}

static PyObject *py_minhash_cuda_init(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  uint32_t dim, seed = static_cast<uint32_t>(time(NULL)), devices = 0;
  uint16_t samples;
  int verbosity = 0;
  static const char *kwlist[] = {
      "dim", "samples", "seed", "devices", "verbosity", NULL
  };

  /* Parse the input tuple */
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "IH|IIi", const_cast<char**>(kwlist), &dim, &samples,
      &seed, &devices, &verbosity)) {
    return NULL;
  }
  MHCUDAResult result = mhcudaSuccess;
  MinhashCudaGenerator *gen;
  Py_BEGIN_ALLOW_THREADS
  gen = mhcuda_init(dim, samples, seed, devices, verbosity, &result);
  Py_END_ALLOW_THREADS
  switch (result) {
    case mhcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to minhash_cuda_init");
      return NULL;
    case mhcudaNoSuchDevice:
      set_cuda_device_error();
      return NULL;
    case mhcudaMemoryAllocationFailure:
      set_cuda_malloc_error();
      return NULL;
    case mhcudaMemoryCopyError:
      set_cuda_memcpy_error();
      return NULL;
    case mhcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "minhash_cuda_init failure (bug?)");
      return NULL;
    case mhcudaSuccess:
      return PyLong_FromUnsignedLongLong(reinterpret_cast<uintptr_t>(gen));
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from minhash_cuda_init");
      return NULL;
  }
}

static PyObject *py_minhash_cuda_retrieve_vars(PyObject *self, PyObject *args) {
  uint64_t gen_ptr;
  if (!PyArg_ParseTuple(args, "K", &gen_ptr)) {
    return NULL;
  }
  MinhashCudaGenerator *gen =
      reinterpret_cast<MinhashCudaGenerator *>(static_cast<uintptr_t>(gen_ptr));
  if (gen == nullptr) {
    PyErr_SetString(PyExc_ValueError, "MinHashCuda Generator pointer is null.");
    return NULL;
  }
  auto params = mhcuda_get_parameters(gen);
  npy_intp dims[] = {params.dim, params.samples, 0};
  auto rs_obj = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(
      2, dims, NPY_FLOAT32, false));
  auto ln_cs_obj = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(
      2, dims, NPY_FLOAT32, false));
  auto betas_obj = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(
      2, dims, NPY_FLOAT32, false));
  auto rs = reinterpret_cast<float *>(PyArray_DATA(rs_obj));
  auto ln_cs = reinterpret_cast<float *>(PyArray_DATA(ln_cs_obj));
  auto betas = reinterpret_cast<float *>(PyArray_DATA(betas_obj));
  int result;
  Py_BEGIN_ALLOW_THREADS
  result = mhcuda_retrieve_random_vars(gen, rs, ln_cs, betas);
  Py_END_ALLOW_THREADS
  switch (result) {
    case mhcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to minhash_cuda_retrieve_vars");
      return NULL;
    case mhcudaNoSuchDevice:
      set_cuda_device_error();
      return NULL;
    case mhcudaMemoryAllocationFailure:
      set_cuda_malloc_error();
      return NULL;
    case mhcudaMemoryCopyError:
      set_cuda_memcpy_error();
      return NULL;
    case mhcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "minhash_cuda_retrieve_vars failure (bug?)");
      return NULL;
    case mhcudaSuccess:
      return Py_BuildValue("OOO", rs_obj, ln_cs_obj, betas_obj);
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from minhash_cuda_retrieve_vars");
      return NULL;
  }
}

static PyObject *py_minhash_cuda_assign_vars(PyObject *self, PyObject *args) {
  PyObject *rs_obj, *ln_cs_obj, *betas_obj;
  uint64_t gen_ptr;
  if (!PyArg_ParseTuple(args, "KOOO", &gen_ptr, &rs_obj, &ln_cs_obj, &betas_obj)) {
    return NULL;
  }
  MinhashCudaGenerator *gen =
      reinterpret_cast<MinhashCudaGenerator*>(static_cast<uintptr_t>(gen_ptr));
  if (gen == nullptr) {
    PyErr_SetString(PyExc_ValueError, "MinHashCuda Generator pointer is null.");
    return NULL;
  }
  auto params = mhcuda_get_parameters(gen);
  int64_t const_size = params.dim * params.samples;
  pyarray rs_arr(PyArray_FROM_OTF(rs_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  if (!rs_arr) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert rs to numpy array");
    return NULL;
  }
  auto size = PyArray_SIZE(rs_arr.get());
  if (size != const_size) {
    PyErr_SetString(PyExc_ValueError, "rs.size must be equal to dim * samples");
    return NULL;
  }
  const float *rs = reinterpret_cast<float *>(PyArray_DATA(rs_arr.get()));
  pyarray ln_cs_arr(PyArray_FROM_OTF(ln_cs_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  if (!ln_cs_arr) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert ln_cs to numpy array");
    return NULL;
  }
  size = PyArray_SIZE(ln_cs_arr.get());
  if (size != const_size) {
    PyErr_SetString(PyExc_ValueError, "ln_cs.size must be equal to dim * samples");
    return NULL;
  }
  const float *ln_cs = reinterpret_cast<float *>(PyArray_DATA(ln_cs_arr.get()));
  pyarray betas_arr(PyArray_FROM_OTF(betas_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  if (betas_arr == NULL) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert betas to numpy array");
    return NULL;
  }
  size = PyArray_SIZE(betas_arr.get());
  if (size != const_size) {
    PyErr_SetString(PyExc_ValueError, "betas.size must be equal to dim * samples");
    return NULL;
  }
  const float *betas = reinterpret_cast<float *>(PyArray_DATA(betas_arr.get()));
  int result;
  Py_BEGIN_ALLOW_THREADS
  result = mhcuda_assign_random_vars(gen, rs, ln_cs, betas);
  Py_END_ALLOW_THREADS
  switch (result) {
    case mhcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to minhash_cuda_assign_vars");
      return NULL;
    case mhcudaNoSuchDevice:
      set_cuda_device_error();
      return NULL;
    case mhcudaMemoryAllocationFailure:
      set_cuda_malloc_error();
      return NULL;
    case mhcudaMemoryCopyError:
      set_cuda_memcpy_error();
      return NULL;
    case mhcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "minhash_cuda_assign_vars failure (bug?)");
      return NULL;
    case mhcudaSuccess:
      Py_RETURN_NONE;
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from minhash_cuda_assign_vars");
      return NULL;
  }
}

static PyObject *py_minhash_cuda_calc(PyObject *self, PyObject *args,
                                      PyObject *kwargs) {
  PyObject *csr_matrix;
  uint64_t gen_ptr;
  uint32_t row_start = 0, row_finish = 0xFFFFFFFFu;

  static const char *kwlist[] = {
      "gen", "matrix", "row_start", "row_finish", NULL
  };

  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "KO|II", const_cast<char**>(kwlist),
      &gen_ptr, &csr_matrix, &row_start, &row_finish)) {
    return NULL;
  }
  MinhashCudaGenerator *gen =
      reinterpret_cast<MinhashCudaGenerator*>(static_cast<uintptr_t>(gen_ptr));
  if (gen == nullptr) {
    PyErr_SetString(PyExc_ValueError, "MinHashCuda Generator pointer is null.");
    return NULL;
  }
  PyObject *scipy = PyImport_ImportModule("scipy.sparse");
  if (scipy == nullptr) {
    PyErr_SetString(PyExc_ImportError, "Failed to import scipy.sparse.csr_matrix");
    return NULL;
  }
  PyObject *matrix_type = PyObject_GetAttrString(scipy, "csr_matrix");
  if (matrix_type == nullptr) {
    PyErr_SetString(PyExc_ImportError, "Failed to import scipy.sparse.csr_matrix");
    return NULL;
  }
  if (!PyObject_TypeCheck(csr_matrix, reinterpret_cast<PyTypeObject *>(matrix_type))) {
    PyErr_SetString(PyExc_TypeError,
                    "The second argument must be of type scipy.sparse.csr_matrix");
    return NULL;
  }
  pyarray weights_obj(PyArray_FROM_OTF(PyObject_GetAttrString(
      csr_matrix, "data"), NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
  if (!weights_obj) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert csr_matrix.data to numpy.array");
    return NULL;
  }
  pyarray cols_obj(PyArray_FROM_OTF(PyObject_GetAttrString(
      csr_matrix, "indices"), NPY_UINT32, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY));
  if (!cols_obj) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert csr_matrix.indices to numpy.array");
    return NULL;
  }
  pyarray rows_obj(PyArray_FROM_OTF(PyObject_GetAttrString(
      csr_matrix, "indptr"), NPY_UINT32, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY));
  if (!rows_obj) {
    PyErr_SetString(PyExc_ValueError, "Failed to convert csr_matrix.indptr to numpy.array");
    return NULL;
  }
  auto rows_dims = PyArray_DIMS(rows_obj.get());
  auto weights = reinterpret_cast<float *>(PyArray_DATA(weights_obj.get()));
  auto cols = reinterpret_cast<uint32_t *>(PyArray_DATA(cols_obj.get()));
  auto rows = reinterpret_cast<uint32_t *>(PyArray_DATA(rows_obj.get()));
  auto length = static_cast<uint32_t>(rows_dims[0]) - 1;

  if (row_start >= length) {
    PyErr_SetString(PyExc_ValueError,
                    "minhash_cuda_calc: row_start must be less than the "
                    "total number of rows in the input matrix.");
    return NULL;
  }
  row_finish = std::min(row_finish, length);
  if (row_finish <= row_start) {
    PyErr_SetString(PyExc_ValueError,
                    "minhash_cuda_calc: row_finish must be greater than "
                    "row_start.");
    return NULL;
  }
  length = row_finish - row_start;
  std::unique_ptr<uint32_t[]> sliced_rows;
  if (row_start > 0) {
    uint32_t offset = rows[row_start];
    weights += offset;
    cols += offset;
    sliced_rows.reset(new uint32_t[length + 1]);
    #pragma omp simd
    for (uint32_t i = row_start; i <= row_finish; i++) {
      sliced_rows[i - row_start] = rows[i] - offset;
    }
  }

  auto params = mhcuda_get_parameters(gen);
  npy_intp hash_dims[] = {length, params.samples, 2, 0};
  auto output_obj = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(
      3, hash_dims, NPY_UINT32, false));
  if (!output_obj) {
    return NULL;
  }
  auto output = reinterpret_cast<uint32_t *>(PyArray_DATA(output_obj));
  int result;
  Py_BEGIN_ALLOW_THREADS
  result = mhcuda_calc(gen, weights, cols,
                       sliced_rows? sliced_rows.get() : rows,
                       length, output);
  Py_END_ALLOW_THREADS
  switch (result) {
    case mhcudaInvalidArguments:
      PyErr_SetString(PyExc_ValueError,
                      "Invalid arguments were passed to minhash_cuda_calc");
      return NULL;
    case mhcudaNoSuchDevice:
      set_cuda_device_error();
      return NULL;
    case mhcudaMemoryAllocationFailure:
      set_cuda_malloc_error();
      return NULL;
    case mhcudaMemoryCopyError:
      set_cuda_memcpy_error();
      return NULL;
    case mhcudaRuntimeError:
      PyErr_SetString(PyExc_AssertionError, "minhash_cuda_calc failure (bug?)");
      return NULL;
    case mhcudaSuccess:
      return reinterpret_cast<PyObject *>(output_obj);
    default:
      PyErr_SetString(PyExc_AssertionError,
                      "Unknown error code returned from minhash_cuda_calc");
      return NULL;
  }
}

static PyObject *py_minhash_cuda_fini(PyObject *self, PyObject *args) {
  uint64_t gen_ptr;
  if (!PyArg_ParseTuple(args, "K", &gen_ptr)) {
    return NULL;
  }
  MinhashCudaGenerator *gen =
      reinterpret_cast<MinhashCudaGenerator*>(static_cast<uintptr_t>(gen_ptr));
  mhcuda_fini(gen);
  Py_RETURN_NONE;
}
