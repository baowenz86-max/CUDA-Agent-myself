import os
import subprocess
from pathlib import Path


# RTX 4060 Laptop GPU
# Ada Lovelace
ARCH = "sm_89"


# 固定编译工具版本，避免受 PATH 和系统默认版本影响
CUDA_HOME = Path("/usr/local/cuda-13.3")
NVCC = str(CUDA_HOME / "bin" / "nvcc")
CUOBJDUMP = str(CUDA_HOME / "bin" / "cuobjdump")
HOST_CC = "/usr/bin/gcc-15"
HOST_CXX = "/usr/bin/g++-15"


KERNEL_DIR = Path("agent_workdir/kernels")


ENV = os.environ.copy()

ENV["CC"] = HOST_CC
ENV["CXX"] = HOST_CXX
ENV["CUDAHOSTCXX"] = HOST_CXX


def run_cmd(cmd):

    print("\n====================")
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


    return result.returncode == 0



def compile_to_cubin(cu_file):

    cubin_file = cu_file.with_suffix(".cubin")


    cmd = [
        NVCC,
        "--expt-relaxed-constexpr",
        f"-arch={ARCH}",
        "-ccbin",
        HOST_CXX,
        "-cubin",
        str(cu_file),
        "-o",
        str(cubin_file)
    ]


    success = run_cmd(cmd)


    if success:

        print(
            f"[PASS] Generated {cubin_file}"
        )
        return cubin_file

    else:

        print(
            f"[FAIL] cubin compile failed: {cu_file}"
        )

        return None



def dump_sass(cubin_file):

    sass_file = cubin_file.with_suffix(".sass")


    cmd = [
        CUOBJDUMP,
        "--dump-sass",
        str(cubin_file)
    ]


    print("\nDump SASS:")


    result = subprocess.run(
        cmd,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )


    print(result.stdout)


    if result.returncode == 0:

        with open(
            sass_file,
            "w"
        ) as f:

            f.write(
                result.stdout
            )


        print(
            f"[PASS] Saved {sass_file}"
        )


    else:

        print(
            "[FAIL] cuobjdump failed"
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

        print(
            f"\nProcessing {cu}"
        )


        cubin = compile_to_cubin(cu)


        if cubin:

            dump_sass(cubin)



if __name__ == "__main__":
    main()
