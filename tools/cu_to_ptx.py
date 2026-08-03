import os
import subprocess
from pathlib import Path


# CUDA架构
# RTX 4060 Laptop = Ada Lovelace = sm_89
ARCH = "sm_89"


# CUDA路径
NVCC = "nvcc"


KERNEL_DIR = Path("agent_workdir/kernels")
ENV = os.environ.copy()

ENV["CC"] = "gcc-13"
ENV["CXX"] = "g++-13"
ENV["CUDAHOSTCXX"] = "/usr/bin/g++-13"

def compile_to_ptx(cu_file):

    ptx_file = cu_file.with_suffix(".ptx")


    cmd = [
        NVCC,
        f"-arch={ARCH}",
        "-ccbin",
        "/usr/bin/g++-13",
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
