"""Configuration for the OpenAI API used by CUDA-Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIAPIConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 300
    max_completion_tokens: int = 32768
    reasoning_effort: str = "medium"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @classmethod
    def from_env(cls) -> "OpenAIAPIConfig":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 OpenAI API 密钥环境变量：OPENAI_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).strip().rstrip("/"),
            model=os.environ.get("CUDA_AGENT_MODEL", "gpt-5.6").strip(),
            timeout_seconds=_positive_int("CUDA_AGENT_API_TIMEOUT", 300),
            max_completion_tokens=_positive_int("CUDA_AGENT_MAX_TOKENS", 32768),
            reasoning_effort=_choice(
                "CUDA_AGENT_REASONING_EFFORT",
                "medium",
                {"none", "low", "medium", "high", "xhigh", "max"},
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
