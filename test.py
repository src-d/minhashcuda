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

if __name__ == "__main__":
    unittest.main()
