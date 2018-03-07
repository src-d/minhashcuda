from multiprocessing import cpu_count
import os
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from shutil import copyfile
from subprocess import check_call
from sys import platform


class CMakeBuild(build_py):
    SHLIBEXT = "dylib" if platform == "darwin" else "so"

    def run(self):
        if not self.dry_run:
            self._build()
        super(CMakeBuild, self).run()

    def get_outputs(self, *args, **kwargs):
        outputs = super(CMakeBuild, self).get_outputs(*args, **kwargs)
        outputs.extend(self._shared_lib)
        return outputs

    def _build(self, builddir=None):
        check_call(("cmake", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCUDA_TOOLKIT_ROOT_DIR=%s" % os.getenv(
                        "CUDA_TOOLKIT_ROOT_DIR",
                        "must_export_CUDA_TOOLKIT_ROOT_DIR"),
                    "."))
        check_call(("make", "-j%d" % cpu_count()))
        self.mkpath(self.build_lib)
        shlib = "libMHCUDA." + self.SHLIBEXT
        dest = os.path.join(self.build_lib, shlib)
        copyfile(shlib, dest)
        self._shared_lib = [dest]


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

setup(
    name="libMHCUDA",
    description="Accelerated Weighted MinHash-ing on GPU",
    version="2.0.3",
    license="Apache Software License",
    author="Vadim Markovtsev",
    author_email="vadim@sourced.tech",
    url="https://github.com/src-d/minhashcuda",
    download_url="https://github.com/src-d/minhashcuda",
    py_modules=["libMHCUDA"],
    install_requires=["numpy"],
    distclass=BinaryDistribution,
    cmdclass={'build_py': CMakeBuild},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ]
)

# python3 setup.py bdist_wheel
# auditwheel repair -w dist dist/*
# twine upload dist/*manylinux*
