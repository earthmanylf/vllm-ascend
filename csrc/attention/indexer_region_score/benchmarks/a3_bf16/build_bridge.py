#!/usr/bin/env python3
"""Compile the benchmark's Torch-to-ACLNN bridge without the vLLM extension."""

import argparse
import os
import subprocess
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import include_paths, library_paths

root = Path(__file__).resolve().parent
sdk = Path(os.environ["ASCEND_HOME_PATH"])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--opapi-dir", type=Path, required=True, help="Directory containing the built libcust_opapi.so")
vendor = parser.parse_args().opapi_dir.resolve()
if not (vendor / "libcust_opapi.so").is_file():
    parser.error("--opapi-dir must contain libcust_opapi.so")
includes = include_paths() + [
    str(Path(torch_npu.__file__).parent / "include"),
    str(sdk / "include"),
    str(sdk / "include/aclnn"),
]
libraries = library_paths() + [str(Path(torch_npu.__file__).parent / "lib"), str(sdk / "lib64"), str(vendor)]
command = [
    "g++",
    "-std=c++17",
    "-O2",
    "-shared",
    "-fPIC",
    "-Wall",
    "-Wextra",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
    str(root / "bridge.cpp"),
]
command += ["-I" + path for path in includes]
command += [flag for path in libraries for flag in ("-L" + path, "-Wl,-rpath," + path)]
command += [
    "-ltorch",
    "-ltorch_cpu",
    "-lc10",
    "-ltorch_npu",
    "-lcust_opapi",
    "-lopapi",
    "-lnnopbase",
    "-lascendcl",
    "-o",
    str(root / "bridge.so"),
]
subprocess.run(command, check=True)
print(root / "bridge.so")
