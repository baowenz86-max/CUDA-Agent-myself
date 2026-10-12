# CUDA-Agent 工作流设计

## 目标

仓库只保留一种自动生成路径：通过 DeepSeek Chat Completions API 生成和修复
CUDA 实现。不再依赖 Codex CLI，也不要求远端模型
访问本地文件或执行命令。

## 当前工作流

```text
CUDA-Agent-Ops-6K
        │
        ▼
读取 task / 写入 model.py
        │
        ▼
固定 seed 采集参考契约 → DeepSeek API 三阶段生成
        │
        ▼
编译 ──失败──► 契约、失败证据与候选历史反馈 ──► 修复并重新编译
        │成功
        ▼
正确性验证 ──失败──► 契约、失败证据与候选历史反馈 ──► 修复并重新编译
        │成功
        ▼
性能分析 ──失败──► 保存正确候选并记录 profiling 失败
        │成功
        ▼
生成 PTX/SASS + 保存 results 快照
```

## 模块职责

| 模块 | 职责 |
|---|---|
| `main.py` | 向后兼容的简短命令行入口 |
| `cuda_agent/workflow.py` | 编排数据加载、生成、修复、验证、性能测试和归档 |
| `cuda_agent/config.py` | DeepSeek API 环境变量的唯一读取和校验入口 |
| `cuda_agent/api.py` | DeepSeek API 请求、响应校验和受限文件写入 |
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

- DeepSeek API：密钥通过 `DEEPSEEK_API_KEY` 提供；端点默认为
  `https://api.deepseek.com`，可用 `DEEPSEEK_BASE_URL` 覆盖。模型默认为
  `deepseek-v4-pro`；推理模式和强度分别由 `CUDA_AGENT_THINKING_MODE`（默认
  `enabled`）和 `CUDA_AGENT_REASONING_EFFORT`（默认 `low`）控制。超时与输出上限
  可通过 `CUDA_AGENT_API_TIMEOUT` 和 `CUDA_AGENT_MAX_TOKENS` 调整。
- CUDA 工具链：`CUDA_HOME`、`CC`、`CXX`、`CUDA_ARCH`。
- 模型只能更新 `model_new.py` 和 `kernels/*.cu|*.cpp`，路径在写入前统一校验。
- API 请求失败、格式错误、编译失败、验证失败和 profiling 失败均有明确状态。

## 后续建议

1. 用 DeepSeek API 做一个最小任务联调，确认 JSON 响应和代码质量。
2. 根据 API 限额调整超时时间、最大 token 数和并发策略。
3. 从多次运行报告中提取统一指标，再增加 CSV/JSON 汇总脚本用于横向比较。

## 正确性修复流程

生成前先在固定 seed 下采集参考契约；之后按 Python、CUDA、binding 三阶段生成。
初始实现按参考 `forward` 的计算步骤拆分，暂不做跨步骤融合。参考契约给出
已安装 PyTorch 的 API 签名、shape/dtype/stride、参数键和逐语句检查点；无法安全
插桩的控制流会明确标记，不改变参考代码，也不阻止正常最终输出验证。

`ModelNew` 保留构造函数、`forward` 的参数和 `state_dict` 层级；参考可追踪时还须
提供 `debug_forward(*inputs)`，依次 `yield (checkpoint_id, value)`，包含最终返回。
元组赋值的检查点必须保留元组结构，包括参考返回的辅助值。禁止算子的限制同样
适用于调试执行。调试接口只定位错误，不能替代正常 `forward` 的五次验证。

每轮修复接收当前最好源码、固定失败输入的信息、第一个错误检查点和近期历史，
并返回如下根因假设（未经执行的假设不能视为事实）：

```json
{
  "analysis": {
    "hypothesis": "输出 grid 被截断，但 kernel 没有 grid-stride loop",
    "evidence": ["错误从后面的 batch 开始", "launcher 将 grid 限制为 65535"],
    "changes": ["增加 grid-stride loop，使每个输出元素都被写入"]
  },
  "files": {"kernels/custom_ops.cu": "完整修改后源码"},
  "delete_files": ["kernels/obsolete_binding.cpp"]
}
```

生成阶段文件范围由代码强制执行；远程模型始终不能修改参考模型、验证器或基础设施。
修复后重新检查接口、编译、严格加载参数，再验证。候选按五次正常验证通过情况、
通过用例数、失败阶段和相同失败用例的误差选择；同等结果保留较早版本，退化候选
会被回退。回退删除新添源码并恢复最好候选，保留原生构建缓存；下一次编译入口校验源码指纹，决定复用二进制或增量构建。

外层每轮完整生成使用干净的生成文件集合；历史最好候选单独保存。默认每轮初始
生成后最多评估 3 个候选（初始版本加两次修复），共最多 3 轮完整生成。

### 诊断资料和重放

每次运行的 `correctness/` 目录保存：

- `task_contract.json`：参考步骤、运行时 API 签名和接口信息。
- `reference_fixture.pt`：固定初始化参数；`failure_inputs.pt`：首个失败输入。
- `verification.json`：正常验证的结构化结果；`diagnostic.json`：独立进程中的中间诊断。
- `api_calls.jsonl`、`history.jsonl`、`repair_context.json`：API 分析、候选结果和回退记录。
- `candidates/NNNN/`：完整生成源码、哈希清单、diff、命令日志及该版本的诊断和重放输入。

在匹配候选源码的工作目录中重新编译，然后使用该候选保存的输入重放：

```bash
bash utils/compile.sh
python -m utils.verification --seed 2026 --replay /absolute/path/correctness/candidates/0001/failure_inputs.pt
python -m utils.verification --seed 2026 --replay /absolute/path/correctness/candidates/0001/failure_inputs.pt --diagnose
```

实际 seed 见 `verification.json`；默认任务 seed 为 `2026 + task_id * 100`，输入 seed
依次为任务 seed 加 1 到 5。只重放本地可信的 `.pt` 文件。重放结果单独写入
`replay_verification.json`，不能覆盖正常验收结果。候选的 `sources/` 只含生成文件；
重放时将其放入具备原参考模型与 `utils/` 的独立工作目录，避免使用其他候选的扩展。

配置新增 `CUDA_AGENT_VERIFICATION_SEED`（基数默认 2026）和
`CUDA_AGENT_COMMAND_TIMEOUT`（外部命令默认 600 秒）。seed 固定随机数据，不保证
所有 GPU 算法逐位确定；设备、PyTorch/CUDA 版本也记录在报告中。

编译默认关闭 `--use_fast_math`，避免近似数学改变正确性；显式设置
`CUDA_AGENT_FAST_MATH=1` 可恢复。生成源码使用 cuBLAS/cuDNN 时链接对应库，目标
环境须提供这些库。此改动不增加性能达标门槛或自动融合优化循环。

### 测试

```bash
python -m unittest discover -v
CUDA_AGENT_GPU_TESTS=1 python -m unittest tests.test_tensor_diagnostics -v
python main.py --task-start 0 --task-end 10
```

无 PyTorch 时，张量回归测试会明确跳过；无 CUDA 时，大输出覆盖的 CUDA 回归测试
会跳过。GPU 回归覆盖 grid 截断与完整写入，其余张量测试覆盖位置参数、分组轴、
卷积索引、histogram 过滤、NaN/Inf、逐步定位及失败输入重放。目标任务重跑前后应
分别比较接口、编译和正确性通过率，不能仅比较 SUCCESS 数量。

### 单 CUDA 文件、调度与增量编译

生成与修复统一使用 `kernels/custom_ops.cu`，其中可以包含多个 kernel 和 launcher。
`.cu` 只包含 CUDA kernel 和原始指针、标量、`cudaStream_t` launcher；Tensor 包装、
检查、分配、stream 获取和 pybind 注册放在 `kernels/*_binding.cpp`。
API 写入前与编译前的源码检查均拒绝 `.cu` 中的 torch/ATen/c10/pybind 依赖及注册，
也拒绝 `.cpp` 中的 CUDA kernel 或 launch 语法。
修复响应可用 `delete_files` 显式删除生成的 kernels 源码；省略文件不会删除。
删除不能与写入冲突，也不能指向固定基础设施或符号链接。仅删除时 `files` 可为空对象。
所有路径和内容先检查，写入或删除失败时回滚整组变更。生成阶段禁止删除。
API 拒绝其他 `.cu` 路径；源码检查和结果保存再次检查 `kernels/` 中恰有这一个
CUDA 文件（包含对子目录的检查）。绑定仍可拆为多个 `.cpp`。

编译和 PTX/SASS 生成使用独立编译名额，不占用 GPU 执行名额。
`CUDA_AGENT_COMPILE_WORKERS` 默认 1，可按 CPU/内存资源调整；
`CUDA_AGENT_GPUS` 指定 GPU 列表，未设置时使用 `CUDA_VISIBLE_DEVICES`，否则默认 `0`。
验证、诊断和 profiling 仍按 GPU 名额互斥。每条命令记录 `resource`、`wait`
和 `execution`：等待时间与实际子进程时间分开，超时从子进程开始执行时计算。

同一次任务运行的候选修复和重新生成保留 `build/forced_compile/` 的 Ninja 缓存。
编译入口按原生源码、工作目录头文件、编译设置、目标架构和工具链计算指纹，并校验
缓存二进制哈希。只改 `model_new.py` 时仍调用编译检查入口，但直接复用 `.so`，
不启动原生编译；原生输入改变时交给 Ninja 增量构建。候选回滚不改写内容相同的源码，
避免无意义地更新文件时间。构建失败移除工作目录的 `.so`，防止验证旧二进制。
重新启动任务仍创建新的隔离工作目录，不跨运行复用缓存。
