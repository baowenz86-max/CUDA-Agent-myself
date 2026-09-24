"""Compile generated CUDA sources and disassemble them to SASS."""

import subprocess
from pathlib import Path

from cuda_agent.tools.toolchain import build_environment, cuda_arch, cuda_home, host_cxx

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO_ROOT / "agent_workdir" / "kernels"


def run_command(command: list[str]) -> tuple[bool, str]:
    print(f"[RUN] {' '.join(command)}")
    result = subprocess.run(
        command, env=build_environment(), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    print(result.stdout)
    return result.returncode == 0, result.stdout


def generate_sass(cu_file: Path) -> bool:
    cubin_file = cu_file.with_suffix(".cubin")
    compiled, _ = run_command([
        str(cuda_home() / "bin/nvcc"), "--expt-relaxed-constexpr",
        f"-arch={cuda_arch()}", "-ccbin", host_cxx(), "-cubin", str(cu_file),
        "-o", str(cubin_file),
    ])
    if not compiled:
        print(f"[FAIL] cubin compile failed: {cu_file}")
        return False

    dumped, sass = run_command([
        str(cuda_home() / "bin/cuobjdump"), "--dump-sass", str(cubin_file)
    ])
    if not dumped:
        print(f"[FAIL] cuobjdump failed: {cubin_file}")
        return False
    sass_file = cubin_file.with_suffix(".sass")
    sass_file.write_text(sass, encoding="utf-8")
    print(f"[PASS] saved {sass_file}")
    return True


def main() -> int:
    sources = sorted(KERNEL_DIR.glob("*.cu"))
    if not sources:
        print(f"No .cu files found in {KERNEL_DIR}")
        return 0
    results = [generate_sass(source) for source in sources]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
