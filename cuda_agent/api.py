"""通过 DeepSeek Chat Completions API 调用远程 CUDA 代码生成模型。"""

import json
import hashlib
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


def _compact(value):
    """Keep concrete evidence while omitting duplicated tracebacks/tensor dumps."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()
                if k not in {'traceback', 'error_excerpt', 'documentation'}
                and not (k == 'message' and value.get('kind') in {'tensor_values', 'tensor_structure'})}
    if isinstance(value, list):
        return [_compact(x) for x in value]
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + ' ... (full evidence retained in the run directory)'
    return value


def _allowed_files(stage):
    return {'python': 'model_new.py', 'cuda': 'kernels/custom_ops.cu',
            'binding': 'kernels/*.cpp'}.get(stage, 'model_new.py, kernels/custom_ops.cu, kernels/*.cpp')


def _example_path(stage):
    return {'cuda': 'kernels/custom_ops.cu', 'binding': 'kernels/custom_ops_binding.cpp'}.get(stage, 'model_new.py')


def _response_metadata(response):
    if not isinstance(response, dict):
        return {}
    choices = response.get('choices')
    choice = choices[0] if isinstance(choices, list) and choices else {}
    return {'finish_reason': choice.get('finish_reason') if isinstance(choice, dict) else None,
            'usage': response.get('usage')}


def _response_contract(data, workdir, run_dir, stage):
    from agent_workdir.utils.operator_contract import validate_contract, contract_call_errors
    if stage == 'python' and 'operator_contract' not in data:
        raise ValueError('Python generation must return operator_contract metadata')
    existing = run_dir / 'custom_ops_contract.json' if run_dir else None
    contract = data.get('operator_contract')
    if contract is None and existing and existing.exists():
        contract = json.loads(existing.read_text())
    if contract is None:
        return None  # legacy direct repairs without a generated contract
    reference_path = run_dir / 'task_contract.json' if run_dir else None
    reference = json.loads(reference_path.read_text()) if reference_path and reference_path.exists() else None
    validate_contract(contract, reference)
    if stage in {'cuda', 'binding'} and 'operator_contract' in data:
        if not existing or not existing.exists() or contract != json.loads(existing.read_text()):
            raise ValueError('CUDA/binding generation cannot change the established operator_contract')
    source = data['files'].get('model_new.py', _read_file(workdir / 'model_new.py'))
    errors = contract_call_errors(source, contract)
    if errors:
        raise ValueError('\n'.join(errors))
    if 'operator_contract' in data:
        if run_dir is None:
            raise ValueError('operator_contract metadata requires a run directory')
        return (run_dir / 'custom_ops_contract.json', contract)
    return None


def _reject_repeated_sources(data, workdir, run_dir):
    from .candidates import source_files
    sources = {str(p.relative_to(workdir)): p.read_text() for p in source_files(workdir)}
    proposed = dict(sources)
    proposed.update(data['files'])
    for name in data.get('delete_files', []):
        proposed.pop(name, None)
    if proposed == sources:
        raise ValueError('Repair has no source changes; submit a different evidence-backed fix')
    path = run_dir / 'repair_context.json' if run_dir else None
    if not path or not path.exists():
        return
    context = json.loads(path.read_text())
    if not context.get('different_hypothesis_required'):
        return
    manifest = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in proposed.items()}
    if any(manifest == r.get('manifest') for r in context.get('recent_candidates', [])):
        raise ValueError('Repair repeats an evaluated source version; use a different testable hypothesis')


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
    if choice.get('finish_reason') == 'length':
        raise ValueError('API response truncated (finish_reason=length); do not apply partial source files')
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

    readonly = ["binding.cpp", "binding_registry.h", "utils/operator_contract.py"]
    extra = "".join(f"\n===== READ ONLY: {name} =====\n{_read_file(workdir / name)}\n" for name in readonly)
    if run_dir is not None:
        for name in ("task_contract.json", "custom_ops_contract.json", "source_checks.json", "interface.json", "verification.json", "diagnostic.json", "repair_context.json"):
            if (run_dir / name).exists():
                data = json.loads(_read_file(run_dir / name))
                extra += f"\n===== READ ONLY: {name} =====\n{json.dumps(_compact(data), ensure_ascii=False)}\n"
    return f"""
===== SKILL.md =====
{_read_file(workdir / "SKILL.md")}

===== model.py =====
{_read_file(workdir / "model.py")}

===== model_new.py =====
{_read_file(workdir / "model_new.py")}

===== 现有 CUDA/C++ kernel 文件 =====
{"".join(kernel_sources) if kernel_sources else "(当前没有 kernel 文件)"}
===== 固定验证规则摘要 =====
先检查源码限制、严格 state_dict/签名与编译后接口，再执行五个固定输入的正常 forward。
类型/容器、shape、dtype、device 必须与参考一致；Tensor 值 atol=1e-2、rtol=1e-2。
debug_forward 仅定位错误，不能替代正常 forward 验证；不得修改验证规则或容限。
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
    if not all(isinstance(name, str) and isinstance(content, str) for name, content in data['files'].items()):
        raise ValueError('files 中的文件名和内容必须都是字符串。')

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


def _apply_files(data: dict, workdir: Path, allowed_stage: str | None = None, *, metadata=None) -> None:
    """Validate the complete response, then atomically apply every file."""
    root = workdir.resolve()
    files = data["files"]

    delete_files = data.get("delete_files", [])
    if not isinstance(delete_files, list) or not all(isinstance(name, str) for name in delete_files):
        raise ValueError("delete_files 必须是文件路径字符串列表。")
    if delete_files and allowed_stage is not None:
        raise ValueError("只有修复阶段允许 delete_files。")
    if not files and not delete_files:
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
        if root not in target.parents or target != root / relative_path:
            raise ValueError(f"拒绝越界路径：{relative_name}")

        validated_files.append((relative_path, target, content))

    deletions = []
    for name in delete_files:
        relative_path = Path(name)
        if len(relative_path.parts) != 2 or relative_path.parts[0] != "kernels" or not _is_allowed_path(relative_path):
            raise ValueError(f"拒绝删除不允许的路径：{name}")
        if relative_path.suffix == ".cu" and relative_path != Path("kernels/custom_ops.cu"):
            raise ValueError(f"拒绝删除不允许的 CUDA 路径：{name}")
        target = (root / relative_path).resolve()
        if root not in target.parents or target != root / relative_path or target in deletions or any(target == item[1] for item in validated_files):
            raise ValueError(f"删除路径越界、重复或与写入冲突：{name}")
        deletions.append(target)

    # Reject misplaced wrappers before proceeding to the next generation stage.
    from .validation import native_structure_errors
    for relative_path, _, content in validated_files:
        errors = native_structure_errors(relative_path.as_posix(), content)
        if errors:
            raise ValueError("\n".join(errors))

    # Trusted orchestrator metadata participates in the same rollback bundle.
    # Models cannot choose its destination or write arbitrary metadata files.
    if metadata is not None:
        target, contract = metadata
        validated_files.append((Path('custom_ops_contract.json'), target,
                                json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False)))

    backups = {target: target.read_bytes() if target.exists() else None for _, target, _ in validated_files}
    backups.update({target: target.read_bytes() if target.exists() else None for target in deletions})
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
            LOGGER.info("API 已更新文件：%s", target.name)
        for target in deletions:
            target.unlink(missing_ok=True)
            LOGGER.info("API 已删除文件：%s", target.relative_to(root))
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
.cu 只能包含 CUDA kernel 与使用原始指针、标量、cudaStream_t 的 launcher，禁止 torch/ATen/c10/pybind 依赖、Tensor 包装与注册。Tensor 检查、分配、当前 stream 获取与 pybind 注册统一放在 kernels/*_binding.cpp；.cpp 不得包含 CUDA kernel 或 launch 语法。
禁止修改 model.py、utils/、binding.cpp、binding_registry.h、SKILL.md。

你的回复必须且只能是合法 JSON，不要使用 Markdown 代码块，不要解释。
格式示例由当前阶段决定，见下文。全部文件内容必须放入 files 的 JSON 字符串中。

files 中只放需要新增或修改的完整文件。省略文件不会删除文件。仅修复阶段可以额外返回 "delete_files": ["kernels/obsolete_binding.cpp"] 显式删除 kernels/ 下生成的源码；不能删除固定基础设施，不能同时写入和删除同一路径。仅删除时 files 可以为空对象。不要返回 shell 命令。修复必须提供 analysis，但其中内容是待验证假设，不能宣称已经运行测试。
""".strip()

    example = {"analysis": {"hypothesis": "根因假设或接口选择", "evidence": ["具体证据"],
                             "changes": ["修改和预期检查结果"]},
               "files": {_example_path(allowed_stage): "该文件的完整内容"}}
    system_prompt += "\n当前阶段允许文件：" + _allowed_files(allowed_stage)
    system_prompt += "\nJSON 格式示例：" + json.dumps(example, ensure_ascii=False)
    if allowed_stage == "python":
        system_prompt += "\n第一阶段还必须返回顶层 operator_contract，具体字段见任务要求。"

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

    started_at = time.monotonic()
    LOGGER.info("调用 DeepSeek API：model=%s stage=%s", config.model, allowed_stage or "repair")
    # Correct response framing locally, without rerunning generation or consuming
    # an algorithm-repair attempt. Transport/I/O failures are not format retries.
    for correction in range(3):
        response_data = None
        content = ""
        try:
            request = Request(config.chat_completions_url,
                              data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {config.api_key}"}, method="POST")
            with urlopen(request, timeout=config.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            content = _response_content(response_data)
            data = _extract_json(content)
            analysis = _validate_analysis(data, require_analysis)
            metadata = _response_contract(data, workdir_path, run_dir, allowed_stage)
            if require_analysis:
                _reject_repeated_sources(data, workdir_path, run_dir)
            _apply_files(data, workdir_path, allowed_stage, metadata=metadata)
            _record_call(run_dir, stage=allowed_stage or "repair", success=True,
                         correction_attempt=correction, analysis=analysis,
                         files=sorted(data["files"]), delete_files=data.get("delete_files", []),
                         elapsed_seconds=time.monotonic() - started_at, **_response_metadata(response_data))
            return True
        except (TimeoutError, HTTPError, URLError, OSError) as exc:
            LOGGER.error("DeepSeek API transport/I/O failure: %s", exc)
            _record_call(run_dir, stage=allowed_stage or "repair", success=False,
                         failure_category="transport_or_io", error=f"{type(exc).__name__}: {exc}",
                         correction_attempt=correction, elapsed_seconds=time.monotonic() - started_at,
                         **_response_metadata(response_data))
            return False
        except (KeyError, IndexError, TypeError, ValueError, SyntaxError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            category = "truncated" if "finish_reason=length" in failure else (
                "json" if isinstance(exc, json.JSONDecodeError) or '没有 JSON 对象' in failure else "response_validation")
            LOGGER.warning("API response rejected (%s): %s", category, failure)
            _record_call(run_dir, stage=allowed_stage or "repair", success=False,
                         failure_category=category, error=failure, correction_attempt=correction,
                         elapsed_seconds=time.monotonic() - started_at, **_response_metadata(response_data))
            if correction == 2:
                return False
            # Keep the original task/context; provide the exact validation error
            # and a small response excerpt instead of growing full-code history.
            payload["messages"] = payload["messages"][:2] + [{"role": "user", "content": (
                "上一响应未应用，原源码保持不变。请纠正响应并重新提交完整合法 JSON。\n"
                f"验证错误：{failure}\n当前阶段：{allowed_stage or 'repair'}\n"
                f"允许文件：{_allowed_files(allowed_stage)}\n"
                "字符串必须正确转义；不得修改不属于当前阶段的文件；不得返回截断内容。\n"
                f"上一响应开头：{content[:1200]}") }]
    return False
