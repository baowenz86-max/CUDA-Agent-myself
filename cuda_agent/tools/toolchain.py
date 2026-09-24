"""Environment-overridable CUDA toolchain settings."""

import os
from pathlib import Path

DEFAULT_CUDA_HOME = "/public/app/cuda/12.6"
DEFAULT_HOST_CC = "/usr/bin/gcc-13"
DEFAULT_HOST_CXX = "/usr/bin/g++-13"
DEFAULT_CUDA_ARCH = "sm_89"  # RTX 4060 / Ada Lovelace


def cuda_home() -> Path:
    return Path(os.environ.get("CUDA_HOME", DEFAULT_CUDA_HOME))


def host_cxx() -> str:
    return os.environ.get("CXX", DEFAULT_HOST_CXX)


def cuda_arch() -> str:
    return os.environ.get("CUDA_ARCH", DEFAULT_CUDA_ARCH)


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    root, cxx = cuda_home(), host_cxx()
    env.update({
        "CUDA_HOME": str(root),
        "CUDACXX": str(root / "bin/nvcc"),
        "CC": os.environ.get("CC", DEFAULT_HOST_CC),
        "CXX": cxx,
        "CUDAHOSTCXX": cxx,
        "PATH": f"{root / 'bin'}{os.pathsep}{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{root / 'lib64'}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}",
    })
    return env
