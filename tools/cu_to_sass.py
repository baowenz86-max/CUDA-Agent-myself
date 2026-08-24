import os
import subprocess
from pathlib import Path


# RTX 4060 Laptop GPU
# Ada Lovelace
ARCH = "sm_89"


NVCC = "nvcc"
CUOBJDUMP = "cuobjdump"


KERNEL_DIR = Path("agent_workdir/kernels")


ENV = os.environ.copy()

# CUDA需要gcc13
ENV["CC"] = "gcc-13"
ENV["CXX"] = "g++-13"
ENV["CUDAHOSTCXX"] = "/usr/bin/g++-13"


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
        "nvcc",
        "--expt-relaxed-constexpr",
        "-arch=sm_89",
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