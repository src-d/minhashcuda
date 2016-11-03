import unittest

from datasketch import WeightedMinHashGenerator
import libMHCUDA
import numpy
from scipy.sparse import csr_matrix


class MHCUDATests(unittest.TestCase):
    def test_calc_tiny(self):
        v1 = [1, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 4]
        v2 = [2, 0, 0, 0, 4, 3, 8, 0, 0, 0, 0, 4, 7, 10, 0, 0, 0, 0, 0, 0, 9, 0, 0]
        bgen = WeightedMinHashGenerator(len(v1))
        gen = libMHCUDA.minhash_cuda_init(len(v1), 128, devices=1, verbosity=2)
        libMHCUDA.minhash_cuda_assign(gen, bgen.rs, bgen.ln_cs, bgen.betas)
        m = csr_matrix(numpy.array([v1, v2], dtype=numpy.float32))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        self.assertEqual(hashes.shape, (2, 128, 2))
        true_hashes = numpy.array([bgen.minhash(v1).hashvalues,
                                   bgen.minhash(v2).hashvalues], dtype=numpy.uint32)
        self.assertEqual(true_hashes.shape, (2, 128, 2))
        try:
            self.assertTrue((hashes == true_hashes).all())
        except AssertionError as e:
            print("---- TRUE ----")
            print(true_hashes)
            print("---- FALSE ----")
            print(hashes)
            raise e from None

    def _test_calc_big(self, devices):
        numpy.random.seed(0)
        data = numpy.random.randint(0, 100, (6400, 130))
        mask = numpy.random.randint(0, 5, data.shape)
        data *= (mask >= 4)
        del mask
        bgen = WeightedMinHashGenerator(data.shape[-1])
        gen = libMHCUDA.minhash_cuda_init(data.shape[-1], 128, devices=devices, verbosity=2)
        libMHCUDA.minhash_cuda_assign(gen, bgen.rs, bgen.ln_cs, bgen.betas)
        m = csr_matrix(data, dtype=numpy.float32)
        print(m.nnz / (m.shape[0] * m.shape[1]))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        self.assertEqual(hashes.shape, (len(data), 128, 2))
        true_hashes = numpy.array([bgen.minhash(line).hashvalues for line in data],
                                  dtype=numpy.uint32)
        self.assertEqual(true_hashes.shape, (len(data), 128, 2))
        try:
            self.assertTrue((hashes == true_hashes).all())
        except AssertionError as e:
            for r in range(hashes.shape[0]):
                if (hashes[r] != true_hashes[r]).any():
                    print("first invalid row:", r)
                    print(hashes[r])
                    print(true_hashes[r])
                    break
            raise e from None

    def test_calc_big(self):
        self._test_calc_big(1)

    def test_calc_big_2gpus(self):
        self._test_calc_big(3)


if __name__ == "__main__":
    unittest.main()
