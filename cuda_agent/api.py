"""通过 DeepSeek Chat Completions API 调用远程 CUDA 代码生成模型。"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cuda_agent.config import DeepSeekAPIConfig


ALLOWED_SUFFIXES = {".cu", ".cpp"}
LOGGER = logging.getLogger(__name__)


def _content_as_text(content: Any) -> str:
    """Normalize common Chat Completions content representations to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part.strip()).strip()
    return ""


def _response_content(response_data: Any) -> str:
    """Extract the assistant's final text or explain why the response is empty."""
    if not isinstance(response_data, Mapping):
        raise ValueError(
            f"API 响应顶层应为 JSON 对象，实际类型为 {type(response_data).__name__}。"
        )

    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            "API 响应没有 choices；顶层字段为 "
            f"{sorted(str(key) for key in response_data.keys())}。"
        )

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("API 响应的 choices[0] 不是 JSON 对象。")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError(
            "API 响应没有 assistant message；choice 字段为 "
            f"{sorted(str(key) for key in choice.keys())}。"
        )

    content = _content_as_text(message.get("content"))
    if content:
        return content

    usage = response_data.get("usage")
    token_usage = "unknown"
    if isinstance(usage, Mapping):
        token_usage = (
            f"prompt={usage.get('prompt_tokens', 'unknown')}, "
            f"completion={usage.get('completion_tokens', 'unknown')}, "
            f"total={usage.get('total_tokens', 'unknown')}"
        )
    refusal = message.get("refusal")
    refusal_status = (
        "present" if isinstance(refusal, str) and refusal.strip() else "none"
    )
    reasoning_status = (
        "present" if _content_as_text(message.get("reasoning_content")) else "none"
    )
    raise ValueError(
        "API 返回了空的 assistant 最终内容；"
        f"finish_reason={choice.get('finish_reason', 'unknown')}, "
        f"usage=({token_usage}), refusal={refusal_status}, "
        f"reasoning_content={reasoning_status}, "
        f"message_fields={sorted(str(key) for key in message.keys())}。"
        "finish_reason=length 表示达到输出 token 上限或上下文限制。DeepSeek 的 "
        "reasoning_content 也占用 max_tokens；可提高 CUDA_AGENT_MAX_TOKENS，"
        "降低 CUDA_AGENT_REASONING_EFFORT，或将 CUDA_AGENT_THINKING_MODE 设为 disabled。"
    )


def _read_file(path: Path) -> str:
    if not path.exists():
        return "(文件不存在)"
    return path.read_text(encoding="utf-8")


def _build_context(workdir: Path, run_dir: Path | None = None) -> str:
    kernel_sources = []

    kernels_dir = workdir / "kernels"
    if kernels_dir.exists():
        for path in sorted(kernels_dir.glob("*")):
            if path.suffix in ALLOWED_SUFFIXES:
                kernel_sources.append(
                    f"\n===== {path.relative_to(workdir)} =====\n{_read_file(path)}"
                )

    readonly = ["binding.cpp", "binding_registry.h", "utils/verification.py", "utils/diagnostics.py", "utils/compile.py"]
    extra = "".join(f"\n===== READ ONLY: {name} =====\n{_read_file(workdir / name)}\n" for name in readonly)
    if run_dir is not None:
        for name in ("task_contract.json", "source_checks.json", "interface.json", "verification.json", "diagnostic.json", "repair_context.json"):
            if (run_dir / name).exists():
                extra += f"\n===== READ ONLY: {name} =====\n{_read_file(run_dir / name)}\n"
    return f"""
===== SKILL.md =====
{_read_file(workdir / "SKILL.md")}

===== model.py =====
{_read_file(workdir / "model.py")}

===== model_new.py =====
{_read_file(workdir / "model_new.py")}

===== 现有 CUDA/C++ kernel 文件 =====
{"".join(kernel_sources) if kernel_sources else "(当前没有 kernel 文件)"}
{extra}
"""


def _extract_json(text: str) -> dict:
    """支持纯 JSON 或 ```json ... ``` 格式的模型回复。"""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("模型回复中没有 JSON 对象。")

    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise ValueError('模型回复必须是 {"files": {...}} 格式。')

    return data


def _is_allowed_path(relative_path: Path) -> bool:
    """只允许模型写 model_new.py 与 kernels 下的 .cu/.cpp 文件。"""
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False

    if relative_path == Path("model_new.py"):
        return True

    return (
        len(relative_path.parts) >= 2
        and relative_path.parts[0] == "kernels"
        and relative_path.suffix in ALLOWED_SUFFIXES
    )


def _apply_files(data: dict, workdir: Path, allowed_stage: str | None = None) -> None:
    """Validate the complete response, then atomically apply every file."""
    root = workdir.resolve()
    files = data["files"]

    if not files:
        raise ValueError("模型没有返回任何待修改文件。")

    validated_files = []
    for relative_name, content in files.items():
        if not isinstance(relative_name, str) or not isinstance(content, str):
            raise ValueError("files 中的文件名和内容必须都是字符串。")

        relative_path = Path(relative_name)
        if len(relative_path.parts) > 2 or not _is_allowed_path(relative_path):
            raise ValueError(f"拒绝写入不允许的路径：{relative_name}")

        if allowed_stage == "python" and relative_path != Path("model_new.py"):
            raise ValueError("Python generation may only write model_new.py")
        if allowed_stage in {"cuda", "binding"}:
            suffix = ".cu" if allowed_stage == "cuda" else ".cpp"
            if len(relative_path.parts) != 2 or relative_path.parts[0] != "kernels" or relative_path.suffix != suffix:
                raise ValueError(f"{allowed_stage} generation may only write kernels/*{suffix}")
        if relative_path.suffix == ".cu" and relative_path != Path("kernels/custom_ops.cu"):
            raise ValueError("CUDA source must be kernels/custom_ops.cu; multiple .cu files are forbidden")
        target = (root / relative_path).resolve()
        if root not in target.parents:
            raise ValueError(f"拒绝越界路径：{relative_name}")

        validated_files.append((relative_path, target, content))

    backups = {target: target.read_bytes() if target.exists() else None for _, target, _ in validated_files}
    temporaries = []
    try:
        # Stage every file before promoting any of them; rollback the bundle on I/O failure.
        for relative_path, target, content in validated_files:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as temp_file:
                temp_file.write(content)
                temporaries.append((Path(temp_file.name), target))
        for temporary, target in temporaries:
            temporary.replace(target)
            LOGGER.info("API 已更新文件：%s", target.relative_to(root))
    except OSError:
        for target, original in backups.items():
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original)
        raise
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)


def _validate_analysis(data: dict, required: bool) -> dict | None:
    analysis = data.get("analysis")
    if analysis is None and not required:
        return None
    if not isinstance(analysis, dict) or not isinstance(analysis.get("hypothesis"), str) or not analysis["hypothesis"].strip():
        raise ValueError("修复回复必须提供 analysis.hypothesis 根因假设")
    for field in ("evidence", "changes"):
        if not isinstance(analysis.get(field), list) or not analysis[field] or not all(isinstance(x, str) and x.strip() for x in analysis[field]):
            raise ValueError(f"修复回复必须提供非空字符串列表 analysis.{field}")
    return analysis


def _record_call(run_dir, **event):
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "api_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_agent(instruction: str, workdir: str, *, allowed_stage: str | None = None,
              run_dir: Path | None = None, require_analysis: bool = False) -> bool:
    """请求远程模型，并将其返回的受限文件变更写入 agent_workdir。"""
    try:
        config = DeepSeekAPIConfig.from_env()
    except ValueError as exc:
        LOGGER.error("DeepSeek API 配置错误：%s", exc)
        _record_call(run_dir, stage=allowed_stage or "repair", success=False, error=str(exc))
        return False

    workdir_path = Path(workdir)

    system_prompt = """
你是 CUDA kernel 优化助手。你不能访问服务器、不能读取本地文件、
不能执行 shell 命令；所有必要上下文已在用户消息中提供。

只允许修改 model_new.py、kernels/custom_ops.cu，或 kernels/ 下的 .cpp 文件。
kernels/ 必须且只能有一个 .cu：custom_ops.cu。全部 CUDA kernel 和 launcher 放入该文件；修复时也不得新增其他 .cu。
禁止修改 model.py、utils/、binding.cpp、binding_registry.h、SKILL.md。

你的回复必须且只能是合法 JSON，不要使用 Markdown 代码块，不要解释。
格式如下：
{
  "analysis": {
    "hypothesis": "待验证的根因假设；生成时说明语义与接口选择",
    "evidence": ["来自参考契约或失败诊断的具体证据"],
    "changes": ["本次修改及预期验证结果"]
  },
  "files": {
    "model_new.py": "该文件的完整内容",
    "kernels/custom_ops.cu": "该文件的完整内容"
  }
}

files 中只放需要新增或修改的完整文件。不要返回 shell 命令。修复必须提供 analysis，但其中内容是待验证假设，不能宣称已经运行测试。
""".strip()

    user_prompt = f"""
任务要求：
{instruction}

以下是工作目录中的文件内容：
{_build_context(workdir_path, run_dir)}
""".strip()

    payload = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "thinking": {"type": config.thinking_mode},
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if config.thinking_mode == "enabled":
        payload["reasoning_effort"] = config.reasoning_effort

    request = Request(
        url=config.chat_completions_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    started_at = time.monotonic()
    LOGGER.info(
        "调用 DeepSeek API：model=%s endpoint=%s max_tokens=%d thinking=%s reasoning_effort=%s",
        config.model,
        request.full_url,
        config.max_tokens,
        config.thinking_mode,
        config.reasoning_effort if config.thinking_mode == "enabled" else "n/a",
    )
    LOGGER.debug("API user prompt (%d chars):\n%s", len(user_prompt), user_prompt)
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        content = _response_content(response_data)

        LOGGER.debug("API response (%d chars):\n%s", len(content), content)
        data = _extract_json(content)
        analysis = _validate_analysis(data, require_analysis)
        _apply_files(data, workdir_path, allowed_stage)
        _record_call(run_dir, stage=allowed_stage or "repair", success=True, analysis=analysis,
                     files=sorted(data["files"]), elapsed_seconds=time.monotonic() - started_at)
        LOGGER.info("DeepSeek API 调用成功，耗时 %.2fs", time.monotonic() - started_at)
        return True

    except TimeoutError as exc:
        failure = f"TimeoutError: {exc}"
        LOGGER.error(
            "DeepSeek API 请求超时（等待 %d 秒仍未收到响应）。",
            config.timeout_seconds,
        )

    except HTTPError as exc:
        failure = f"HTTPError: {exc}"
        body = exc.read().decode("utf-8", errors="replace")
        LOGGER.error("DeepSeek API HTTP 错误：%s\n%s", exc.code, body)
    except URLError as exc:
        failure = f"URLError: {exc}"
        LOGGER.error("无法连接 DeepSeek API：%s", exc.reason)
    except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        LOGGER.error("DeepSeek API 回复或文件变更格式错误：%s", exc)

    _record_call(run_dir, stage=allowed_stage or "repair", success=False, error=failure,
                 elapsed_seconds=time.monotonic() - started_at)
    return False
