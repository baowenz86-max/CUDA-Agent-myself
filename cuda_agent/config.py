"""Configuration for the school-provided model API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SchoolAPIConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 300
    max_tokens: int = 32768
    thinking_mode: str = "disabled"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @classmethod
    def from_env(cls) -> "SchoolAPIConfig":
        values = {
            "CUDA_AGENT_API_KEY": os.environ.get("CUDA_AGENT_API_KEY", "").strip(),
            "CUDA_AGENT_BASE_URL": os.environ.get("CUDA_AGENT_BASE_URL", "").strip(),
            "CUDA_AGENT_MODEL": os.environ.get("CUDA_AGENT_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"缺少学校 API 环境变量：{', '.join(missing)}")
        return cls(
            api_key=values["CUDA_AGENT_API_KEY"],
            base_url=values["CUDA_AGENT_BASE_URL"],
            model=values["CUDA_AGENT_MODEL"],
            timeout_seconds=_positive_int("CUDA_AGENT_API_TIMEOUT", 300),
            max_tokens=_positive_int("CUDA_AGENT_MAX_TOKENS", 32768),
            thinking_mode=_choice(
                "CUDA_AGENT_THINKING_MODE", "disabled", {"enabled", "disabled"}
            ),
        )


def _positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是正整数，当前值：{raw_value!r}") from exc
    if value < 1:
        raise ValueError(f"{name} 必须是正整数，当前值：{raw_value!r}")
    return value


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 必须是 {allowed} 之一，当前值：{value!r}")
    return value
