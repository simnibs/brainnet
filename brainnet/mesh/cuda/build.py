import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension, BuildExtension

from pathlib import Path

have_cuda = torch.cuda.is_available() and CUDA_HOME is not None
if not have_cuda:
    print("ERROR: cuda is not available.")
    exit()

# cd into this folder then execute
#
#   python build.py build_ext --inplace

# CUDA_HOME = env base directory

cpp_std = 17 # 20

# base_dir = Path("/mrhome/jesperdn/repositories/SuperSynth")
# src_dir = base_dir / "ext" / "graph" / "cuda"
src_dir = Path(__file__).parent.resolve()

include_dirs = []

# only CONDA_PREFIX/lib is included in runtime dirs
runtime_library_dirs = [torch.utils.cpp_extension.library_paths(cuda=True)]

sources = ["extensions.cpp", "nearest_neighbor.cu", "self_intersections.cu"]
sources = [src_dir / f for f in sources]

# nvcc
macros = [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]

nvcc_args = [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]
if os.name != "nt":
    nvcc_args.append(f"-std=c++{cpp_std}")

nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
if nvcc_flags_env != "":
    nvcc_args.extend(nvcc_flags_env.split(" "))

# gcc
cpp_args = [f"-std=c++{cpp_std}"]

extra_compile_args = dict(cxx=cpp_args, nvcc=nvcc_args)

# cuda include
cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
    prefix = os.environ.get("CONDA_PREFIX", None)
    assert prefix is not None
    prefix = Path(prefix)
    cub_home = prefix / "include"
    cub_dir = cub_home / "cub"
    assert cub_dir.is_dir()

include_dirs.append(cub_home)

ext_modules = [
    CUDAExtension(
        "extensions",
        sources=[str(s) for s in sources],
        include_dirs=[str(d) for d in include_dirs],
        runtime_library_dirs=runtime_library_dirs,
        define_macros=macros,
        extra_compile_args=extra_compile_args,
    )
]

setup(name="brainnet", ext_modules=ext_modules, cmdclass={"build_ext": BuildExtension})
