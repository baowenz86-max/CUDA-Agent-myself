import os
import subprocess
from pathlib import Path


# CUDA架构
# RTX 4060 Laptop = Ada Lovelace = sm_80
ARCH = "sm_80"


# 固定编译工具版本，避免受 PATH 和系统默认版本影响
CUDA_HOME = Path("/public/app/cuda/13.0")
NVCC = str(CUDA_HOME / "bin" / "nvcc")
HOST_CC = "/usr/bin/gcc"
HOST_CXX = "/usr/bin/g++"


KERNEL_DIR = Path("agent_workdir/kernels")
ENV = os.environ.copy()

ENV["CC"] = HOST_CC
ENV["CXX"] = HOST_CXX
ENV["CUDAHOSTCXX"] = HOST_CXX

def compile_to_ptx(cu_file):

    ptx_file = cu_file.with_suffix(".ptx")


    cmd = [
        NVCC,
        f"-arch={ARCH}",
        "-ccbin",
        HOST_CXX,
        "-ptx",
        str(cu_file),
        "-o",
        str(ptx_file)
    ]


    print("\n====================")
    print("Compile:")
    print(" ".join(cmd))
    print("====================")


    result = subprocess.run(
        cmd,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )


    print(result.stdout)


    if result.returncode == 0:

        print(
            f"[PASS] {ptx_file}"
        )

    else:

        print(
            f"[FAIL] {cu_file}"
        )



def main():

    cu_files = list(
        KERNEL_DIR.glob("*.cu")
    )


    if not cu_files:

        print(
            "No .cu files found"
        )
        return


    for cu in cu_files:

        compile_to_ptx(cu)



if __name__ == "__main__":
    main()
