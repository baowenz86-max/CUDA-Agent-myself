"""End-to-end CUDA kernel generation workflow."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from cuda_agent.api import run_agent
from cuda_agent.log_setup import configure_logging
from cuda_agent.report import WorkflowReport
from cuda_agent.tools.toolchain import build_environment

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover - optional in development environments
    Dataset = None
    load_dataset = None


DATASET_NAME = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = REPO_ROOT / "agent_workdir"
LOCAL_DATASET = (
    REPO_ROOT / "datasets/BytedTsinghua-SIA___cuda-agent-ops-6_k/default/0.0.0"
    / "44a734c78c947bfcba5189cbfd13f57a6d29a698"
    / "cuda-agent-ops-6_k-train.arrow"
)
COMPILE_COMMAND = ("bash", "utils/compile.sh")
VERIFY_COMMAND = (sys.executable, "-m", "utils.verification")
PROFILE_COMMAND = (sys.executable, "-m", "utils.profiling")
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowConfig:
    workdir: Path = DEFAULT_WORKDIR
    generation_attempts: int = 5
    repair_attempts: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, compile, verify, and profile a CUDA kernel."
    )
    parser.add_argument("--task-id", type=int, default=0, help="Dataset row index")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--generation-attempts", type=int, default=5)
    parser.add_argument("--repair-attempts", type=int, default=3)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--verbose", action="store_true", help="Show debug logs in console")
    return parser.parse_args()


def load_task(task_id: int) -> Mapping[str, object]:
    if task_id < 0:
        raise ValueError("task-id must be non-negative")
    if Dataset is None or load_dataset is None:
        raise RuntimeError("The 'datasets' package is required. Install it first.")

    LOGGER.info("加载数据集任务 %d", task_id)
    if LOCAL_DATASET.exists():
        LOGGER.info("使用本地 Arrow 数据集：%s", LOCAL_DATASET)
        dataset = Dataset.from_file(str(LOCAL_DATASET))
    else:
        LOGGER.info("从 Hugging Face Hub 加载 %s", DATASET_NAME)
        dataset = load_dataset(DATASET_NAME)["train"]
    if task_id >= len(dataset):
        raise IndexError(f"task-id {task_id} is out of range (size: {len(dataset)})")

    sample = dataset[task_id]
    LOGGER.info("任务 operators：%s", sample["ops"])
    return sample


def prepare_model(sample: Mapping[str, object], workdir: Path) -> None:
    code = sample.get("code")
    if not isinstance(code, str):
        raise ValueError("Dataset sample has no string 'code' field")
    (workdir / "model.py").write_text(code, encoding="utf-8")
    LOGGER.info("model.py 已生成")


def reset_generated_workspace(workdir: Path) -> None:
    """Remove artifacts produced by the previous task."""
    kernel_dir = workdir / "kernels"
    generated_suffixes = {".cu", ".cpp", ".ptx", ".cubin", ".sass"}
    if kernel_dir.exists():
        for path in kernel_dir.iterdir():
            if path.is_file() and path.suffix in generated_suffixes:
                path.unlink()
                LOGGER.debug("清理旧文件：%s", path)
    for filename in ("model_new.py", "cuda_extension.so"):
        path = workdir / filename
        if path.exists():
            path.unlink()
            LOGGER.debug("清理旧文件：%s", path)


def run_command(
    command: Sequence[str], workdir: Path, env: Mapping[str, str]
) -> tuple[bool, str]:
    """Run one command without a shell and capture its combined output."""
    LOGGER.info("执行命令：%s", " ".join(command))
    result = subprocess.run(
        list(command), cwd=workdir, env=dict(env), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.stdout.strip():
        LOGGER.info("命令输出：\n%s", result.stdout.rstrip())
    LOGGER.info("命令结束：exit_code=%d", result.returncode)
    return result.returncode == 0, result.stdout


def generate_cuda(workdir: Path) -> bool:
    return run_agent(
        """阅读 SKILL.md 和 model.py，生成 CUDA 优化版本。
只修改 model_new.py 和 kernels/ 下的 CUDA/C++ 文件；不要修改 utils/、
binding.cpp 或 binding_registry.h。实现必须通过编译和正确性验证。""",
        str(workdir),
    )


def send_feedback(error_log: str, workdir: Path, stage: str) -> bool:
    return run_agent(
        f"""CUDA 实现的{stage}阶段失败。请分析日志并修复 model_new.py 或 kernels/。
不要修改 utils/、binding.cpp 或 binding_registry.h。

错误日志：
----------------
{error_log}
----------------""",
        str(workdir),
    )


def run_stage_with_repairs(
    name: str,
    command: Sequence[str],
    config: WorkflowConfig,
    env: Mapping[str, str],
) -> tuple[bool, str]:
    """Run a stage and request a model repair between failed attempts."""
    last_log = ""
    for attempt in range(1, config.repair_attempts + 1):
        LOGGER.info("%s 尝试 %d/%d", name, attempt, config.repair_attempts)
        success, last_log = run_command(command, config.workdir, env)
        if success:
            LOGGER.info("%s 成功", name)
            return True, last_log
        if attempt < config.repair_attempts and not send_feedback(
            last_log, config.workdir, name
        ):
            LOGGER.error("%s 的 API 修复请求失败", name)
            break
    return False, last_log


def generate_kernel_artifacts(env: Mapping[str, str]) -> None:
    """Generate optional PTX and SASS artifacts after a successful run."""
    for module_name in (
        "cuda_agent.tools.cu_to_ptx",
        "cuda_agent.tools.cu_to_sass",
    ):
        success, _ = run_command(
            (sys.executable, "-m", module_name), REPO_ROOT, env
        )
        if not success:
            LOGGER.warning("%s 失败；继续保存其他结果", module_name)


def save_result(task_id: int, workdir: Path, logs: Mapping[str, str]) -> Path:
    """Save a complete snapshot of a successful task."""
    save_dir = REPO_ROOT / "results" / f"task_{task_id:04d}"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)

    for filename in ("model.py", "model_new.py"):
        source = workdir / filename
        if source.exists():
            shutil.copy2(source, save_dir / filename)
    kernel_source = workdir / "kernels"
    if kernel_source.exists():
        shutil.copytree(kernel_source, save_dir / "kernels")
    for name, content in logs.items():
        (save_dir / name).write_text(content, encoding="utf-8")
    (save_dir / "meta.txt").write_text(
        f"task_id: {task_id}\ntime: {datetime.now().isoformat()}\nstatus: success\n",
        encoding="utf-8",
    )
    LOGGER.info("结果已保存至 %s", save_dir)
    return save_dir


def execute_workflow(
    task_id: int, config: WorkflowConfig, report: WorkflowReport
) -> bool:
    env = build_environment()
    sample = load_task(task_id)
    report.operators = str(sample.get("ops", "unknown"))
    reset_generated_workspace(config.workdir)
    prepare_model(sample, config.workdir)

    for attempt in range(1, config.generation_attempts + 1):
        LOGGER.info("生成尝试 %d/%d", attempt, config.generation_attempts)
        started = time.monotonic()
        generated = generate_cuda(config.workdir)
        report.add_stage(
            f"generation #{attempt}", generated, time.monotonic() - started,
            "school API",
        )
        if not generated:
            continue
        started = time.monotonic()
        compiled, compile_log = run_stage_with_repairs(
            "compile", COMPILE_COMMAND, config, env
        )
        report.add_stage("compile", compiled, time.monotonic() - started)
        if not compiled:
            continue
        started = time.monotonic()
        verified, verify_log = run_stage_with_repairs(
            "verification", VERIFY_COMMAND, config, env
        )
        report.add_stage("verification", verified, time.monotonic() - started)
        if not verified:
            continue
        started = time.monotonic()
        profiled, profile_log = run_command(PROFILE_COMMAND, config.workdir, env)
        report.add_stage("profiling", profiled, time.monotonic() - started)
        if not profiled:
            LOGGER.error("profiling 失败；本轮不会标记为成功")
            continue

        started = time.monotonic()
        generate_kernel_artifacts(env)
        report.add_stage("artifacts", True, time.monotonic() - started, "best effort")
        save_result(task_id, config.workdir, {
            "compile.log": compile_log,
            "verification.log": verify_log,
            "profile.log": profile_log,
        })
        return True
    return False


def main() -> int:
    args = parse_args()
    config = WorkflowConfig(
        args.workdir.resolve(), args.generation_attempts, args.repair_attempts
    )
    if config.generation_attempts < 1 or config.repair_attempts < 1:
        raise ValueError("Attempt counts must be at least 1")
    log_path = configure_logging(args.log_dir.resolve(), args.task_id, args.verbose)
    report = WorkflowReport(args.task_id, log_path)
    LOGGER.info("启动 CUDA Agent；详细日志：%s", log_path)

    success = False
    error = ""
    try:
        success = execute_workflow(args.task_id, config, report)
        if not success:
            error = "All generation attempts were exhausted. See workflow.log."
    except Exception as exc:  # preserve a report for unexpected failures
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("工作流异常终止")
    finally:
        report_path = report.write(success, error)
        LOGGER.info("工作流报告：%s", report_path)

    if success:
        LOGGER.info("任务成功")
        return 0
    LOGGER.error("任务失败：%s", error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
