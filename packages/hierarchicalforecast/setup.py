import glob
import platform
import subprocess

import setuptools
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, naive_recompile

extra_compile_args = []
extra_link_args = []

# OpenMP support + optimization flags
if platform.system() == "Linux":
    extra_compile_args += ["-fopenmp", "-O2", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    extra_link_args += ["-fopenmp"]
elif platform.system() == "Darwin":
    # Homebrew installs libomp headers/libs outside the default search paths
    try:
        libomp_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
        extra_compile_args += [f"-I{libomp_prefix}/include"]
        extra_link_args += [f"-L{libomp_prefix}/lib"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    extra_compile_args += ["-Xpreprocessor", "-fopenmp", "-O2", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    extra_link_args += ["-lomp"]
elif platform.system() == "Windows":
    extra_compile_args += ["/openmp", "/O2", "/fp:fast", "/DNDEBUG"]

ext_modules = [
    Pybind11Extension(
        name="hierarchicalforecast._lib",
        sources=glob.glob("src/*.cpp"),
        include_dirs=["external_libs/eigen"],
        cxx_std=20,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

ParallelCompile(
    "CMAKE_BUILD_PARALLEL_LEVEL", needs_recompile=naive_recompile
).install()

setuptools.setup(ext_modules=ext_modules)
