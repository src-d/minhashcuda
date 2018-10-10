from multiprocessing import cpu_count
import os
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from shutil import copyfile
from subprocess import check_call
import sys
import sysconfig

with open("README.md") as f:
    long_description = f.read()

def get_python_library():
    """Get path to the python library associated with the current python
    interpreter."""
    cfgvar = sysconfig.get_config_var
    libname = cfgvar("LDLIBRARY")
    python_library = os.path.join(
        cfgvar("LIBDIR") + (cfgvar("multiarchsubdir") or ""),
        libname)
    if os.path.exists(python_library):
        return python_library
    for root, dirnames, filenames in os.walk(cfgvar("base")):
        for filename in filenames:
            if filename == libname:
                return os.path.join(root, filename)
    raise FileNotFoundError(libname)


class CMakeBuild(build_py):
    SHLIBEXT = "dylib" if sys.platform == "darwin" else "so"

    def run(self):
        if not self.dry_run:
            self._build()
        super(CMakeBuild, self).run()

    def get_outputs(self, *args, **kwargs):
        outputs = super(CMakeBuild, self).get_outputs(*args, **kwargs)
        outputs.extend(self._shared_lib)
        return outputs

    def _build(self, builddir=None):
        syspaths = sysconfig.get_paths()
        check_call(("cmake", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCUDA_TOOLKIT_ROOT_DIR=%s" % os.getenv(
                        "CUDA_TOOLKIT_ROOT_DIR",
                        "must_export_CUDA_TOOLKIT_ROOT_DIR"),
                    "-DPYTHON_DEFAULT_EXECUTABLE=python3",
                    "-DPYTHON_INCLUDE_DIRS=" + syspaths["include"],
                    "-DPYTHON_EXECUTABLE=" + sys.executable,
                    "-DPYTHON_LIBRARY=" + get_python_library(),
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="2.1.1",
    license="Apache Software License",
    author="Vadim Markovtsev",
    author_email="vadim@sourced.tech",
    url="https://github.com/src-d/minhashcuda",
    download_url="https://github.com/src-d/minhashcuda",
    py_modules=["libMHCUDA"],
    install_requires=["numpy"],
    distclass=BinaryDistribution,
    cmdclass={"build_py": CMakeBuild},
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
