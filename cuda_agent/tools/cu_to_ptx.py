"""Compile all generated CUDA sources to PTX."""

import subprocess
from pathlib import Path

from cuda_agent.tools.toolchain import build_environment, cuda_arch, cuda_home, host_cxx

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "agent_workdir" / "kernels"


def compile_to_ptx(cu_file: Path) -> bool:
    output = cu_file.with_suffix(".ptx")
    command = [str(cuda_home() / "bin/nvcc"), f"-arch={cuda_arch()}",
               "-ccbin", host_cxx(), "-ptx", str(cu_file), "-o", str(output)]
    print(f"[RUN] {' '.join(command)}")
    result = subprocess.run(
        command, env=build_environment(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    print(result.stdout)
    print(f"[{'PASS' if result.returncode == 0 else 'FAIL'}] {cu_file}")
    return result.returncode == 0


def main() -> int:
    sources = sorted(KERNEL_DIR.glob("*.cu"))
    if not sources:
        print(f"No .cu files found in {KERNEL_DIR}")
        return 0
    results = [compile_to_ptx(source) for source in sources]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
