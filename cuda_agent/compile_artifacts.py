"""Archive task-local Inductor/Triton caches without importing torch."""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

LOGGER = logging.getLogger(__name__)


def prepare_compile_cache(parent: Path, env: Mapping[str, str]) -> tuple[Path, dict[str, str]]:
    """Use a fresh cache for profiling, including when tasks run concurrently."""
    root = Path(tempfile.mkdtemp(prefix="torch_compile_", dir=parent)).resolve()
    profile_env = dict(env)
    profile_env.update(
        TORCHINDUCTOR_CACHE_DIR=str(root / "inductor"),
        TRITON_CACHE_DIR=str(root / "triton"),
        TRITON_STORE_BINARY_ONLY="0",
    )
    return root, profile_env


def save_compile_artifacts(cache: Path, destination: Path, env: Mapping[str, str]) -> None:
    """Keep cache paths and companion IR/source files; disassemble GPU binaries."""
    destination.mkdir(parents=True, exist_ok=True)
    errors = []
    for source in sorted(cache.rglob("*")):
        if not source.is_file() or source.is_symlink() or source.suffix == ".lock":
            continue
        target = destination / source.relative_to(cache)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            errors.append(f"copy {source.relative_to(cache)}: {exc}")

    cuda_bin = Path(env.get("CUDA_HOME", "")) / "bin" / "cuobjdump"
    tool = str(cuda_bin) if cuda_bin.is_file() else shutil.which("cuobjdump", path=env.get("PATH"))
    for binary in sorted(destination.rglob("*.cubin")):
        if not tool:
            errors.append(f"{binary.relative_to(destination)}: cuobjdump unavailable; SASS not exported")
            continue
        try:
            result = subprocess.run(
                [tool, "--dump-sass", str(binary)], env=dict(env), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60,
            )
            if result.returncode:
                errors.append(f"{binary.relative_to(destination)}: cuobjdump exit {result.returncode}: {result.stderr[-4000:]}")
            else:
                binary.with_suffix(".sass").write_text(result.stdout, encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{binary.relative_to(destination)}: {exc}")

    files = sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file())
    manifest = {"files": files, "errors": errors,
                "note": "Only artifacts produced by this profiling run are included; some backends/operators produce no PTX, cubin or shared library."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for error in errors:
        LOGGER.warning("torch.compile artifact: %s", error)
