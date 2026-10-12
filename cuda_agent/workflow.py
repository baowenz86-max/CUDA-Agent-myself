"""End-to-end CUDA kernel generation workflow."""

from __future__ import annotations

import json
import argparse #解析命令行参数
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import logging #记录日志
import os
import shutil #复制、删除目录
import subprocess #执行编译、验证等外部命令
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from cuda_agent.api import run_agent
from cuda_agent.candidates import CandidateStore, clean_sources, read_json
from cuda_agent.validation import check_sources
from cuda_agent.log_setup import CURRENT_TASK_ID, configure_logging
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
DEFAULT_GENERATION_ATTEMPTS = 3
LOGGER = logging.getLogger(__name__)
_LOGGING_SETUP_LOCK = threading.Lock()


# Generation is intentionally split into three calls.  Each call has one
# owner file, so the model can concentrate on one interface boundary and the
# next call can inspect the already-written result.
GENERATION_PROMPTS = (
    (
        "model_new.py",
        """你负责第一阶段：只生成 model_new.py，不要生成或修改任何 .cu/.cpp 文件。
请阅读只读 task_contract.json、验证器、绑定基础设施、SKILL.md 和 model.py；以实际参考代码而非 ops 标注为准，完整梳理参考实现的输入输出 shape、dtype、device、广播、边界条件和数值语义。
先保证与 model.py 语义等价、接口可用。本轮逐参考计算步骤实现，不做跨步骤融合；必须保持 state_dict 键和 forward 参数一致。task_contract.json 可追踪时必须提供 debug_forward(*inputs)，按 checkpoints 顺序 yield (id, value)，包括最终返回。正常 forward 仍返回参考结果，调试接口不能改变计算公式。model_new.py 只能调用后续 CUDA 源码中会实现的自定义算子；请明确、统一地设计算子名称、参数顺序、返回值结构和 dtype/shape 约定，避免伪代码、TODO、占位实现或未定义调用。
保留非 CUDA 或特殊输入路径。禁止使用 torch 算子实现本应下沉到自定义扩展的计算。
只返回 model_new.py 的完整内容；不要返回 kernels 文件，不要修改 model.py、utils/、binding.cpp、binding_registry.h 或 SKILL.md。""",
    ),
    (
        "kernels/*.cu",
        """你负责第二阶段：只生成或修改 kernels/ 下的 CUDA .cu 文件，不要生成或修改 .cpp 或 model_new.py。
先阅读 SKILL.md、model.py、当前 model_new.py 以及现有 kernels 文件，严格以 model_new.py 已确定的自定义算子名称、参数顺序、返回结构和 dtype/shape 约定为接口契约；不要反过来改 Python 接口。
只实现参考契约实际使用的 dtype/device，不额外添加未经验证的 half/double 路径。按参考步骤实现完整、可编译的 CUDA kernel 和 host launcher，覆盖 model_new.py 的所有调用路径。重点核对维度/stride、边界和空张量、非连续输入、dtype 分派、数值精度、同步、线程索引和 launch 参数；不得留下伪代码、TODO、占位实现或未定义符号。需要导出给绑定层的 launcher 必须有清晰且稳定的声明，供下一阶段的 binding.cpp 调用。
只返回需要写入的 .cu 文件完整内容；不要返回 .cpp 文件，不要修改 model_new.py、model.py、utils/、binding.cpp、binding_registry.h 或 SKILL.md。""",
    ),
    (
        "kernels/*.cpp",
        """你负责第三阶段：只生成或修改 kernels/ 下的绑定 .cpp 文件，不要生成或修改 .cu 或 model_new.py。
先阅读 SKILL.md、model.py、model_new.py 以及当前 kernels/*.cu，严格对齐 Python 调用接口和 CUDA launcher 的函数声明。为每个自定义算子提供完整的 PyTorch tensor 参数检查、必要的 dtype/device/shape 处理、正确的 launcher 调用、返回值构造和 REGISTER_BINDING 注册；不得在 .cpp 中实现 CUDA kernel，不得调用 torch:: 算子完成计算。
逐项核对注册名、参数顺序、include、命名空间、声明、返回结构和 dtype 分派，确保 binding.cpp 能独立编译并与 .cu 正确链接。不要留下伪代码、TODO、占位实现或未定义符号。
只返回需要写入的 .cpp 文件完整内容；不要返回 .cu 文件，不要修改 model_new.py、model.py、utils/、binding.cpp、binding_registry.h 或 SKILL.md。""",
    ),
)

REPAIR_PROMPTS = {
    "compile": """这是一次编译失败修复。请把现有源码视为主版本，先逐行分析编译日志，定位第一个真正的错误及其根因，再做最小必要修改。
重点检查：头文件/声明、命名空间、模板和 dtype 分派、CUDA API 类型、kernel launch 参数、host/device 限定符、链接/注册符号以及 nvcc 支持的语法。不要因为日志末尾的级联错误而盲目修改无关代码，也不要重写已经正确的算法。
修复后必须再次核对 model_new.py 与 kernels/ 的接口一致性。仅允许修改 model_new.py 或 kernels/ 下的 .cu/.cpp，禁止修改固定基础设施和编译脚本。""",
    "verification": """这是一次正确性验证失败修复。请先根据验证日志确定失败的是哪一个输入、哪个输出、哪种 dtype/shape 或哪条执行路径，再追踪对应 kernel 的索引和数学语义。
重点排查：维度/stride 与线性索引、广播和 reduction 边界、线程间同步、越界访问、初始化、mask、in-place 副作用、float/half 精度转换、NaN/Inf 处理以及 Python 返回值结构。修复目标是与 model.py 的参考结果严格一致；不要为了掩盖误差而放宽验证，也不要在没有证据时改动性能无关代码。
修复后确认编译接口仍完整，并只修改 model_new.py 或 kernels/ 下的 .cu/.cpp，禁止修改固定基础设施和验证脚本。""",
}


@dataclass(frozen=True)
class WorkflowConfig:
    workdir: Path = DEFAULT_WORKDIR
    generation_attempts: int = DEFAULT_GENERATION_ATTEMPTS
    repair_attempts: int = 3


class GPUPool:
    """A small GPU semaphore; CUDA commands receive one visible GPU at a time."""

    def __init__(self, gpu_ids: Sequence[str]) -> None:
        if not gpu_ids:
            raise ValueError("至少需要一张 GPU")
        self._semaphore = threading.Semaphore(len(gpu_ids))
        self._available = list(gpu_ids)
        self._lock = threading.Lock()

    def acquire(self):
        self._semaphore.acquire()
        with self._lock:
            gpu_id = self._available.pop()
        return _GPULease(self, gpu_id)

    def release(self, gpu_id: str) -> None:
        with self._lock:
            self._available.append(gpu_id)
        self._semaphore.release()


class _GPULease:
    def __init__(self, pool: GPUPool, gpu_id: str) -> None:
        self.pool, self.gpu_id = pool, gpu_id

    def __enter__(self) -> str:
        return self.gpu_id

    def __exit__(self, *_: object) -> None:
        self.pool.release(self.gpu_id)


def gpu_ids_from_environment() -> list[str]:
    configured = os.environ.get("CUDA_AGENT_GPUS", "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        return [item.strip() for item in visible.split(",") if item.strip()]
    return ["0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, compile, verify, and profile CUDA kernels for a task range."
    )
    parser.add_argument(
        "--task-start",
        type=int,
        required=True,
        help="First dataset row index to process (inclusive)",
    )
    parser.add_argument(
        "--task-end",
        type=int,
        required=True,
        help="Last dataset row index to process (inclusive)",
    )
    return parser.parse_args()


def task_ids_from_args(args: argparse.Namespace) -> range:
    """Return the inclusive task ID range requested by the CLI."""
    task_start = getattr(args, "task_start", None)
    task_end = getattr(args, "task_end", None)
    # Keep this helper tolerant of legacy programmatic callers.  The public
    # CLI itself only accepts the two required range arguments.
    legacy_task_id = getattr(args, "task_id", None)
    if legacy_task_id is not None and task_start is not None:
        raise ValueError("--task-id 不能与 --task-start/--task-end 一起使用")
    if task_start is None or task_end is None:
        task_id = 0 if legacy_task_id is None else legacy_task_id
        if task_id < 0:
            raise ValueError("Task IDs must be non-negative")
        return range(task_id, task_id + 1)
    if task_start < 0 or task_end < 0:
        raise ValueError("Task IDs must be non-negative")
    if task_start > task_end:
        raise ValueError("--task-start 不能大于 --task-end")
    return range(task_start, task_end + 1)


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
    command: Sequence[str], workdir: Path, env: Mapping[str, str],
    gpu_pool: GPUPool | None = None,
) -> tuple[bool, str]:
    """Run one command without a shell and capture its combined output."""
    LOGGER.info("执行命令：%s", " ".join(command))
    lease = gpu_pool.acquire() if gpu_pool is not None else nullcontext(None)
    with lease as gpu_id:
        command_env = dict(env)
        if gpu_id is not None:
            command_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        try:
            result = subprocess.run(
                list(command), cwd=workdir, env=command_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                timeout=int(command_env.get("CUDA_AGENT_COMMAND_TIMEOUT", "600")),
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            LOGGER.error("命令超时：%s", " ".join(command))
            return False, output + "\nCommand timed out; the GPU lease has been released."
    if result.stdout.strip():
        LOGGER.info("命令输出：\n%s", result.stdout.rstrip())
    LOGGER.info("命令结束：exit_code=%d", result.returncode)
    return result.returncode == 0, result.stdout


def generate_cuda(workdir: Path, run_dir: Path | None = None) -> bool:
    """Generate Python, CUDA, and binding sources in separate API calls."""
    for stage, (target, prompt) in enumerate(GENERATION_PROMPTS, start=1):
        LOGGER.info("生成阶段 %d/3：%s", stage, target)
        if not run_agent(prompt, str(workdir), allowed_stage=("python", "cuda", "binding")[stage - 1], run_dir=run_dir):
            LOGGER.error("生成阶段 %d 失败：%s", stage, target)
            return False
    return True


def send_feedback(error_log: str, workdir: Path, stage: str, run_dir: Path | None = None) -> bool:
    repair_prompt = REPAIR_PROMPTS.get(stage)
    if repair_prompt is None:
        repair_prompt = (
            "这是一次工作流阶段失败修复。请仅依据现有日志定位根因，保持正确代码不变，"
            "并只修改允许的生成文件。"
        )
    return run_agent(
        f"""{repair_prompt}

当前失败阶段：{stage}
当前源码可能已恢复为最好候选。请结合 repair_context.json 区分当前主版本与被拒绝的修改，
保留正确部分，按第一个错误检查点做最小修改。若连续两轮没有改善，必须提出不同且可验证的根因。
只支持本任务实际使用的 dtype。不得修改诊断/验证器、放宽容限或以填零掩盖未写输出。
回复必须含 analysis.hypothesis、analysis.evidence 和 analysis.changes；这是待验证假设。

完整错误日志（优先处理最早出现的根因）：
----------------
{error_log}
----------------""",
        str(workdir), run_dir=run_dir, require_analysis=True,
    )


def run_stage_with_repairs(
    name: str,
    command: Sequence[str],
    config: WorkflowConfig,
    env: Mapping[str, str],
    before_retry: Callable[[], tuple[bool, str]] | None = None,
    gpu_pool: GPUPool | None = None,
) -> tuple[bool, str]:
    """Run a stage, optionally preparing dependent stages after each repair."""
    last_log = ""
    for attempt in range(1, config.repair_attempts + 1):
        LOGGER.info("%s 尝试 %d/%d", name, attempt, config.repair_attempts)
        success, last_log = run_command(command, config.workdir, env, gpu_pool)
        if success:
            LOGGER.info("%s 成功", name)
            return True, last_log
        if attempt < config.repair_attempts and not send_feedback(
            last_log, config.workdir, name
        ):
            LOGGER.error("%s 的 API 修复请求失败", name)
            break
        if attempt < config.repair_attempts and before_retry is not None:
            prepared, preparation_log = before_retry()
            if not prepared:
                LOGGER.error("%s 修复后的前置步骤失败", name)
                if preparation_log.strip():
                    last_log += (
                        "\n\n修复后、重试该阶段前的准备步骤失败：\n"
                        f"{preparation_log}"
                    )
                break
    return False, last_log


def generate_kernel_artifacts(
    env: Mapping[str, str], workdir: Path, gpu_pool: GPUPool | None = None
) -> None:
    """Generate optional PTX and SASS artifacts after a successful run."""
    artifact_env = dict(env)
    artifact_env["CUDA_AGENT_WORKDIR"] = str(workdir)
    artifact_env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + artifact_env.get("PYTHONPATH", "")
    for module_name in (
        "cuda_agent.tools.cu_to_ptx",
        "cuda_agent.tools.cu_to_sass",
    ):
        success, _ = run_command(
            (sys.executable, "-m", module_name), workdir, artifact_env, gpu_pool
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
    task_id: int, config: WorkflowConfig, report: WorkflowReport,
    gpu_pool: GPUPool,
) -> bool:
    env = build_environment()
    run_dir = report.log_path.parent / "correctness"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(os.environ.get("CUDA_AGENT_VERIFICATION_SEED", "2026")) + task_id * 100
    env.update(CUDA_AGENT_RUN_DIR=str(run_dir.resolve()), CUDA_AGENT_VERIFICATION_SEED=str(seed))
    store = CandidateStore(config.workdir, run_dir)
    sample = load_task(task_id)
    report.operators = str(sample.get("ops", "unknown"))
    reset_generated_workspace(config.workdir)
    clean_sources(config.workdir)
    prepare_model(sample, config.workdir)
    started = time.monotonic()
    ready, reference_log = run_command((sys.executable, "-m", "utils.preflight"), config.workdir, env, gpu_pool)
    report.add_stage("reference contract", ready, time.monotonic() - started)
    if not ready:
        report.failure_reason = "Reference preflight failed: " + reference_log[-6000:]
        return False

    def evaluate():
        # Reports belong to a specific source version; never reuse stale results.
        for name in ("interface.json", "verification.json", "diagnostic.json"):
            (run_dir / name).unlink(missing_ok=True)
        checks = check_sources(config.workdir)
        (run_dir / "source_checks.json").write_text(json.dumps(checks.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        for warning in checks.warnings:
            LOGGER.warning("源代码诊断：%s", warning)
        if checks.errors:
            return "sources", False, "\n".join(checks.errors), None, None
        compiled, compile_log = run_command(COMPILE_COMMAND, config.workdir, env, gpu_pool)
        if not compiled:
            return "compile", False, compile_log, None, None
        interface_ok, interface_log = run_command((sys.executable, "-m", "utils.preflight", "--compiled"), config.workdir, env, gpu_pool)
        if not interface_ok:
            return "interface", False, compile_log + "\n" + interface_log, None, None
        verified, verify_log = run_command(VERIFY_COMMAND, config.workdir, env, gpu_pool)
        verification = read_json(run_dir / "verification.json", {})
        # Exit code alone cannot establish that all fixed normal-forward checks ran.
        rows = verification.get("checks", [])
        verified = verified and verification.get("status") == "passed" and len(rows) == 5 and all(row.get("status") == "passed" for row in rows)
        diagnostic = None
        if not verified and (run_dir / "failure_inputs.pt").exists() and verification.get("first_failure"):
            _, diagnostic_log = run_command((*VERIFY_COMMAND, "--diagnose"), config.workdir, env, gpu_pool)
            diagnostic = read_json(run_dir / "diagnostic.json", {"status": "unavailable", "reason": diagnostic_log[-6000:]})
            sanitizer = shutil.which("compute-sanitizer", path=env.get("PATH"))
            failure_text = verify_log.lower()
            if sanitizer and any(word in failure_text for word in ("illegal memory", "launch failure", "out of bounds")):
                _, sanitizer_log = run_command((sanitizer, "--tool", "memcheck", "--error-exitcode", "86", *VERIFY_COMMAND,
                                                "--replay", str(run_dir / "failure_inputs.pt")), config.workdir, env, gpu_pool)
                diagnostic["sanitizer_log"] = sanitizer_log[-12000:]
                (run_dir / "diagnostic.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        return "verification", verified, compile_log + "\n" + interface_log + "\n" + verify_log, verification, diagnostic

    for generation in range(1, config.generation_attempts + 1):
        clean_sources(config.workdir)
        for name in ("source_checks.json", "interface.json", "verification.json", "diagnostic.json"):
            (run_dir / name).unlink(missing_ok=True)
        started = time.monotonic()
        generated = generate_cuda(config.workdir, run_dir)
        report.add_stage(f"generation #{generation}", generated, time.monotonic() - started, "DeepSeek API")
        if not generated:
            store.capture("sources", False, "Generation failed; see workflow.log and api_calls.jsonl")
            continue
        for repair in range(config.repair_attempts):
            started = time.monotonic()
            stage, success, log, verification, diagnostic = evaluate()
            record = store.capture(stage, success, log, verification, diagnostic)
            report.best_candidate = str(Path(store.best["snapshot"]).relative_to(report.log_path.parent))
            report.add_stage(f"{stage} candidate #{record['id']}", success, time.monotonic() - started, record["decision"])
            if success:
                started = time.monotonic()
                profiled, profile_log = run_command(PROFILE_COMMAND, config.workdir, env, gpu_pool)
                report.add_stage("profiling", profiled, time.monotonic() - started)
                if not profiled:
                    report.failure_reason = "Correctness passed, but profiling failed: " + profile_log[-6000:]
                    return False
                started = time.monotonic()
                generate_kernel_artifacts(env, config.workdir, gpu_pool)
                report.add_stage("artifacts", True, time.monotonic() - started, "best effort")
                save_result(task_id, config.workdir, {"verification.log": log, "profile.log": profile_log})
                return True
            # Restoring always invalidates the binary; evaluate compiles the next repair.
            store.restore_best()
            if repair + 1 == config.repair_attempts:
                break
            if not send_feedback(log, config.workdir, stage if stage == "compile" else "verification", run_dir):
                report.add_stage("repair API", False, 0, "No changes applied; best candidate retained")
                break
    store.restore_best()
    report.failure_reason = store.best_failure()
    return False


def run_task(
    task_id: int, config: WorkflowConfig, args: argparse.Namespace,
    gpu_pool: GPUPool,
) -> bool:
    """Run one task and always write its report before returning."""
    task_token = CURRENT_TASK_ID.set(task_id)
    # configure_logging changes process-wide handlers, so at least make the
    # setup itself atomic while task workers start.
    with _LOGGING_SETUP_LOCK:
        log_path = configure_logging(DEFAULT_LOG_DIR, task_id, False)
    report = WorkflowReport(task_id, log_path)
    LOGGER.info("启动 CUDA Agent；工作目录：%s；详细日志：%s", config.workdir, log_path)

    success = False
    error = ""
    try:
        success = execute_workflow(task_id, config, report, gpu_pool)
        if not success:
            error = report.failure_reason or "All generation attempts were exhausted. See workflow.log."
    except Exception as exc:  # preserve a report for unexpected failures
        error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("工作流异常终止")
    finally:
        report_path = report.write(success, error)
        LOGGER.info("工作流报告：%s", report_path)
        CURRENT_TASK_ID.reset(task_token)

    if success:
        LOGGER.info("任务 %d 成功", task_id)
    else:
        LOGGER.error("任务 %d 失败：%s", task_id, error)
    return success


def main() -> int:
    args = parse_args()
    task_ids = task_ids_from_args(args)
    config = WorkflowConfig()
    if config.generation_attempts < 1 or config.repair_attempts < 1:
        raise ValueError("Attempt counts must be at least 1")

    task_id_list = list(task_ids)
    gpu_pool = GPUPool(gpu_ids_from_environment())
    generation_workers = min(
        len(task_id_list),
        max(1, int(os.environ.get("CUDA_AGENT_GENERATION_WORKERS", len(task_id_list)))),
    )

    def worker(task_id: int) -> bool:
        task_workdir = REPO_ROOT / ".task_workdirs" / f"task_{task_id:04d}"
        if task_workdir.exists():
            shutil.rmtree(task_workdir)
        shutil.copytree(config.workdir, task_workdir)
        task_config = WorkflowConfig(
            task_workdir, config.generation_attempts, config.repair_attempts
        )
        return run_task(task_id, task_config, args, gpu_pool)

    successful_task_ids: list[int] = []
    failed_task_ids: list[int] = []
    with ThreadPoolExecutor(max_workers=generation_workers) as executor:
        futures = {executor.submit(worker, task_id): task_id for task_id in task_id_list}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                success = future.result()
            except Exception:
                LOGGER.exception("任务 %d 工作线程异常", task_id)
                success = False
            (successful_task_ids if success else failed_task_ids).append(task_id)

    LOGGER.info(
        "任务批次完成：成功 %d，失败 %d", len(successful_task_ids), len(failed_task_ids)
    )
    if failed_task_ids:
        LOGGER.error("失败任务：%s", ", ".join(map(str, failed_task_ids)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
