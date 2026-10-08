"""Configuration for DeepSeek's OpenAI-compatible API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekAPIConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 300
    max_tokens: int = 32768
    thinking_mode: str = "enabled"
    reasoning_effort: str = "low"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @classmethod
    def from_env(cls) -> "DeepSeekAPIConfig":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("缺少 DeepSeek API 密钥环境变量：DEEPSEEK_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).strip().rstrip("/"),
            model=os.environ.get("CUDA_AGENT_MODEL", "deepseek-v4-pro").strip(),
            timeout_seconds=_positive_int("CUDA_AGENT_API_TIMEOUT", 300),
            max_tokens=_positive_int("CUDA_AGENT_MAX_TOKENS", 32768),
            thinking_mode=_choice(
                "CUDA_AGENT_THINKING_MODE", "enabled", {"enabled", "disabled"}
            ),
            reasoning_effort=_choice(
                "CUDA_AGENT_REASONING_EFFORT",
                "low",
                {"low", "high", "max"},
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
