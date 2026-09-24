# CUDA-Agent 工作流设计

## 目标

仓库只保留一种自动生成路径：通过学校提供的 OpenAI-compatible Chat
Completions API 生成和修复 CUDA 实现。不再依赖 Codex CLI，也不要求远端模型
访问本地文件或执行命令。

## 当前工作流

```text
CUDA-Agent-Ops-6K
        │
        ▼
读取 task / 写入 model.py
        │
        ▼
学校 API 生成 model_new.py + kernels/*
        │
        ▼
编译 ──失败──► 错误日志反馈给学校 API ──► 修复
        │成功
        ▼
正确性验证 ──失败──► 错误日志反馈给学校 API ──► 修复
        │成功
        ▼
性能分析 ──失败──► 重新进入下一轮生成
        │成功
        ▼
生成 PTX/SASS + 保存 results 快照
```

## 模块职责

| 模块 | 职责 |
|---|---|
| `main.py` | 向后兼容的简短命令行入口 |
| `cuda_agent/workflow.py` | 编排数据加载、生成、修复、验证、性能测试和归档 |
| `cuda_agent/config.py` | 学校 API 环境变量的唯一读取和校验入口 |
| `cuda_agent/api.py` | 学校 API 请求、响应校验和受限文件写入 |
| `cuda_agent/log_setup.py` | 控制台日志和每次运行的 DEBUG 文件日志 |
| `cuda_agent/report.py` | 汇总阶段状态、耗时和失败原因 |
| `cuda_agent/tools/` | CUDA 工具链配置和 PTX/SASS 生成工具 |
| `scripts/` | 数据下载等仓库维护命令 |

## 日志和报告

每次运行使用独立目录，避免覆盖历史记录：

```text
logs/task_0000/20260924_120000_000000/
├── workflow.log
└── workflow_report.md
```

`workflow.log` 包含完整命令输出、API 请求阶段、原始模型回复、重试过程和异常
堆栈。API key 不会写入日志。`workflow_report.md` 面向复盘，包含任务、算子、
总体状态、各阶段耗时和失败摘要。

## 配置边界

- 学校 API：`CUDA_AGENT_API_KEY`、`CUDA_AGENT_BASE_URL`、
  `CUDA_AGENT_MODEL`；超时与输出上限可通过 `CUDA_AGENT_API_TIMEOUT` 和
  `CUDA_AGENT_MAX_TOKENS` 调整。
- CUDA 工具链：`CUDA_HOME`、`CC`、`CXX`、`CUDA_ARCH`。
- 模型只能更新 `model_new.py` 和 `kernels/*.cu|*.cpp`，路径在写入前统一校验。
- API 请求失败、格式错误、编译失败、验证失败和 profiling 失败均有明确状态。

## 后续建议

1. 用学校 API 的真实模型做一个最小任务联调，确认其 Chat Completions 响应格式。
2. 根据学校网关限制调整超时时间、最大 token 数和并发策略。
3. 从多次运行报告中提取统一指标，再增加 CSV/JSON 汇总脚本用于横向比较。
