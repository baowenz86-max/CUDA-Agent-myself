"""通过 OpenAI 兼容 Chat Completions API 调用远程 CUDA 代码生成模型。"""

import json
import logging
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cuda_agent.config import SchoolAPIConfig


ALLOWED_SUFFIXES = {".cu", ".cpp"}
LOGGER = logging.getLogger(__name__)


def _read_file(path: Path) -> str:
    if not path.exists():
        return "(文件不存在)"
    return path.read_text(encoding="utf-8")


def _build_context(workdir: Path) -> str:
    kernel_sources = []

    kernels_dir = workdir / "kernels"
    if kernels_dir.exists():
        for path in sorted(kernels_dir.glob("*")):
            if path.suffix in ALLOWED_SUFFIXES:
                kernel_sources.append(
                    f"\n===== {path.relative_to(workdir)} =====\n{_read_file(path)}"
                )

    return f"""
===== SKILL.md =====
{_read_file(workdir / "SKILL.md")}

===== model.py =====
{_read_file(workdir / "model.py")}

===== model_new.py =====
{_read_file(workdir / "model_new.py")}

===== 现有 CUDA/C++ kernel 文件 =====
{"".join(kernel_sources) if kernel_sources else "(当前没有 kernel 文件)"}
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


def _apply_files(data: dict, workdir: Path) -> None:
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
        if not _is_allowed_path(relative_path):
            raise ValueError(f"拒绝写入不允许的路径：{relative_name}")

        target = (root / relative_path).resolve()
        if root not in target.parents:
            raise ValueError(f"拒绝越界路径：{relative_name}")

        validated_files.append((relative_path, target, content))

    for relative_path, target, content in validated_files:
        target.parent.mkdir(parents=True, exist_ok=True)

        # 同目录临时文件保证 os.replace 在同一文件系统上原子执行。
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        temp_path.replace(target)

        LOGGER.info("API 已更新文件：%s", relative_path)


def run_agent(instruction: str, workdir: str) -> bool:
    """请求远程模型，并将其返回的受限文件变更写入 agent_workdir。"""
    try:
        config = SchoolAPIConfig.from_env()
    except ValueError as exc:
        LOGGER.error("学校 API 配置错误：%s", exc)
        return False

    workdir_path = Path(workdir)

    system_prompt = """
你是 CUDA kernel 优化助手。你不能访问服务器、不能读取本地文件、
不能执行 shell 命令；所有必要上下文已在用户消息中提供。

只允许修改 model_new.py，或 kernels/ 下的 .cu 与 .cpp 文件。
禁止修改 model.py、utils/、binding.cpp、binding_registry.h、SKILL.md。

你的回复必须且只能是合法 JSON，不要使用 Markdown 代码块，不要解释。
格式如下：
{
  "files": {
    "model_new.py": "该文件的完整内容",
    "kernels/example.cu": "该文件的完整内容"
  }
}

files 中只放需要新增或修改的完整文件。不要返回 shell 命令。
""".strip()

    user_prompt = f"""
任务要求：
{instruction}

以下是工作目录中的文件内容：
{_build_context(workdir_path)}
""".strip()

    payload = {
        "model": config.model,
        "temperature": 0.1,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

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
    LOGGER.info("调用学校 API：model=%s endpoint=%s", config.model, request.full_url)
    LOGGER.debug("API user prompt (%d chars):\n%s", len(user_prompt), user_prompt)
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        content = response_data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("API 返回内容为空。")

        LOGGER.debug("API response (%d chars):\n%s", len(content), content)
        _apply_files(_extract_json(content), workdir_path)
        LOGGER.info("学校 API 调用成功，耗时 %.2fs", time.monotonic() - started_at)
        return True

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        LOGGER.error("学校 API HTTP 错误：%s\n%s", exc.code, body)
    except URLError as exc:
        LOGGER.error("无法连接学校 API：%s", exc.reason)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.exception("学校 API 回复或文件变更格式错误：%s", exc)

    return False
