from time import time
import unittest

from datasketch import WeightedMinHashGenerator, WeightedMinHash
import libMHCUDA
import numpy
from scipy.sparse import csr_matrix
from scipy.stats import gamma, uniform


class MHCUDATests(unittest.TestCase):
    def test_calc_tiny(self):
        v1 = [1, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 4]
        v2 = [2, 0, 0, 0, 4, 3, 8, 0, 0, 0, 0, 4, 7, 10, 0, 0, 0, 0, 0, 0, 9, 0, 0]
        bgen = WeightedMinHashGenerator(len(v1))
        gen = libMHCUDA.minhash_cuda_init(len(v1), 128, devices=1, verbosity=2)
        libMHCUDA.minhash_cuda_assign_vars(gen, bgen.rs, bgen.ln_cs, bgen.betas)
        m = csr_matrix(numpy.array([v1, v2], dtype=numpy.float32))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        libMHCUDA.minhash_cuda_fini(gen)
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
        libMHCUDA.minhash_cuda_assign_vars(gen, bgen.rs, bgen.ln_cs, bgen.betas)
        m = csr_matrix(data, dtype=numpy.float32)
        print(m.nnz / (m.shape[0] * m.shape[1]))
        ts = time()
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        print("libMHCUDA:", time() - ts)
        libMHCUDA.minhash_cuda_fini(gen)
        self.assertEqual(hashes.shape, (len(data), 128, 2))
        ts = time()
        true_hashes = numpy.array([bgen.minhash(line).hashvalues for line in data],
                                  dtype=numpy.uint32)
        print("datasketch:", time() - ts)
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

    def test_random_vars(self):
        gen = libMHCUDA.minhash_cuda_init(1000, 128, devices=1, verbosity=2)
        rs, ln_cs, betas = libMHCUDA.minhash_cuda_retrieve_vars(gen)
        libMHCUDA.minhash_cuda_fini(gen)
        self.assertEqual(rs.shape, (128, 1000))
        self.assertEqual(ln_cs.shape, (128, 1000))
        self.assertEqual(betas.shape, (128, 1000))
        cs = numpy.exp(ln_cs)
        a, loc, scale = gamma.fit(rs)
        self.assertTrue(1.97 < a < 2.03)
        self.assertTrue(-0.01 < loc < 0.01)
        self.assertTrue(0.98 < scale < 1.02)
        a, loc, scale = gamma.fit(cs)
        self.assertTrue(1.97 < a < 2.03)
        self.assertTrue(-0.01 < loc < 0.01)
        self.assertTrue(0.98 < scale < 1.02)
        bmin, bmax = uniform.fit(betas)
        self.assertTrue(0 <= bmin < 0.001)
        self.assertTrue(0.999 <= bmax <= 1)

    def test_integration(self):
        numpy.random.seed(1)
        data = numpy.random.randint(0, 100, (6400, 130))
        mask = numpy.random.randint(0, 5, data.shape)
        data *= (mask >= 4)
        del mask
        gen = libMHCUDA.minhash_cuda_init(data.shape[-1], 128, seed=1, verbosity=1)
        m = csr_matrix(data, dtype=numpy.float32)
        print(m.nnz / (m.shape[0] * m.shape[1]))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        libMHCUDA.minhash_cuda_fini(gen)
        self.assertEqual(hashes.shape, (len(data), 128, 2))
        h1 = WeightedMinHash(0, hashes[0])
        h2 = WeightedMinHash(0, hashes[1])
        cudamh = h1.jaccard(h2)
        print(cudamh)
        truemh = numpy.amin(data[:2], axis=0).sum() / numpy.amax(data[:2], axis=0).sum()
        print(truemh)
        self.assertTrue(abs(truemh - cudamh) < 0.005)

    def test_slice(self):
        numpy.random.seed(0)
        data = numpy.random.randint(0, 100, (6400, 130))
        mask = numpy.random.randint(0, 5, data.shape)
        data *= (mask >= 4)
        del mask
        gen = libMHCUDA.minhash_cuda_init(data.shape[-1], 128, verbosity=2)
        m = csr_matrix(data, dtype=numpy.float32)
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        hashes2 = libMHCUDA.minhash_cuda_calc(
            gen, m, row_start=3200, row_finish=4800)
        libMHCUDA.minhash_cuda_fini(gen)
        self.assertTrue((hashes[3200:4800] == hashes2).all())

    def test_backwards(self):
        v1 = [1, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 4]
        v2 = [2, 0, 0, 0, 4, 3, 8, 0, 0, 0, 0, 4, 7, 10, 0, 0, 0, 0, 0, 0, 9, 0, 0]
        gen = libMHCUDA.minhash_cuda_init(len(v1), 128, devices=1, verbosity=2)
        rs, ln_cs, betas = libMHCUDA.minhash_cuda_retrieve_vars(gen)
        bgen = WeightedMinHashGenerator.__new__(WeightedMinHashGenerator)
        bgen.dim = len(v1)
        bgen.rs = rs
        bgen.ln_cs = ln_cs
        bgen.betas = betas
        bgen.sample_size = 128
        bgen.seed = None
        m = csr_matrix(numpy.array([v1, v2], dtype=numpy.float32))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        libMHCUDA.minhash_cuda_fini(gen)
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

    def test_deferred(self):
        v1 = [1, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0, 9, 10, 4]
        v2 = [2, 0, 0, 0, 4, 3, 8, 0, 0, 0, 0, 4, 7, 10, 0, 0, 0, 0, 0, 0, 9, 0, 0]
        gen = libMHCUDA.minhash_cuda_init(len(v1), 128, devices=1, verbosity=2)
        vars = libMHCUDA.minhash_cuda_retrieve_vars(gen)
        libMHCUDA.minhash_cuda_fini(gen)
        gen = libMHCUDA.minhash_cuda_init(
            len(v1), 128, devices=1, deferred=True, verbosity=2)
        libMHCUDA.minhash_cuda_assign_vars(gen, *vars)
        bgen = WeightedMinHashGenerator.__new__(WeightedMinHashGenerator)
        bgen.dim = len(v1)
        bgen.rs, bgen.ln_cs, bgen.betas = vars
        bgen.sample_size = 128
        bgen.seed = None
        m = csr_matrix(numpy.array([v1, v2], dtype=numpy.float32))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m)
        libMHCUDA.minhash_cuda_fini(gen)
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

    def test_float(self):
        v1 = [
            0,          1.0497366,  0.8494359,  0.66231006, 0.66231006, 0.8494359,
            0,          0.66231006, 0.33652836, 0,           0,         0.5359344,
            0.8494359,  0.66231006, 1.0497366,  0.33652836, 0.66231006, 0.8494359,
            0.6800841,  0.33652836]
        gen = libMHCUDA.minhash_cuda_init(len(v1), 128, devices=1, seed=7, verbosity=2)
        vars = libMHCUDA.minhash_cuda_retrieve_vars(gen)
        bgen = WeightedMinHashGenerator.__new__(WeightedMinHashGenerator)
        bgen.dim = len(v1)
        bgen.rs, bgen.ln_cs, bgen.betas = vars
        bgen.sample_size = 128
        bgen.seed = None
        m = csr_matrix(numpy.array(v1, dtype=numpy.float32))
        hashes = libMHCUDA.minhash_cuda_calc(gen, m).astype(numpy.int32)
        libMHCUDA.minhash_cuda_fini(gen)
        self.assertEqual(hashes.shape, (1, 128, 2))
        true_hashes = numpy.array([bgen.minhash(v1).hashvalues], dtype=numpy.int32)
        self.assertEqual(true_hashes.shape, (1, 128, 2))
        try:
            self.assertTrue((hashes == true_hashes).all())
        except AssertionError as e:
            print("---- TRUE ----")
            print(true_hashes)
            print("---- FALSE ----")
            print(hashes)
            raise e from None

    def test_split(self):
        def run_test(v):
            k = sum([len(part) for part in v])
            bgen = WeightedMinHashGenerator(len(k))
            gen = libMHCUDA.minhash_cuda_init(len(k), 128, devices=4, verbosity=2)
            libMHCUDA.minhash_cuda_assign_vars(gen, bgen.rs, bgen.ln_cs, bgen.betas)
            m = csr_matrix(numpy.array(v, dtype=numpy.float32))
            hashes = None
            try:
                hashes = libMHCUDA.minhash_cuda_calc(gen, m)
            finally:
                self.assertIsNotNone(hashes)
                self.assertEqual(hashes.shape, (1, 128, 2))
                libMHCUDA.minhash_cuda_fini(gen)
        # here we try to break minhashcuda with unbalanced partitions
        run_test([[2], [1], [1], [1]])
        run_test([[1] * 50, [1], [1], [1]])
        run_test([[1], [1] * 50, [1], [1]])
        run_test([[1], [1], [1] * 50, [1]])
        run_test([[1], [1], [1], [1] * 50])
        run_test([[1] * 3, [1] * 10, [1] * 5, [1] * 2])


if __name__ == "__main__":
    unittest.main()
