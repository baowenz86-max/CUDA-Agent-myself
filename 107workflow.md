# CUDA-Agent A100 环境配置工作日志

日期：2026-09-01  
平台节点：`anode19`  
项目路径：`~/projects/CUDA-Agent`

## 1. GPU 与驱动确认

在计算节点执行 `nvidia-smi`，确认已分配：

- GPU：NVIDIA A100-SXM4-80GB
- 显存：80 GB
- 驱动版本：580.173.02
- 驱动支持的最高 CUDA 版本：13.0
- GPU 计算能力：8.0

结论：项目的 CUDA 编译目标应设为 `sm_80`，对应环境变量：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
```

## 2. 平台软件环境检查

执行了以下检查命令：

```bash
uv --version
python3 --version

command -v module
module avail cuda
module avail gcc

which nvcc
nvcc --version

which gcc
gcc --version
which g++
g++ --version
```

确认结果：

- uv：`0.11.28`
- Python：`3.12.13`
- 平台提供 CUDA：`cuda/12.6`、`cuda/13.0`
- 初始默认 nvcc：`/public/app/cuda/13.0/bin/nvcc`
- GCC：`/usr/bin/gcc`，版本 `13.3.0`
- G++：`/usr/bin/g++`，版本 `13.3.0`

## 3. 选择并加载编译工具链

考虑到仓库主编译流程原先使用 CUDA 12.6，选择 CUDA 12.6 与系统 GCC/G++ 13.3 作为统一工具链。

已执行：

```bash
cd ~/projects/CUDA-Agent

module unload cuda/13.0
module load cuda/12.6
```

并配置编译环境变量：

```bash
export CUDA_HOME=/public/app/cuda/12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
```

随后确认 `nvcc --version` 为：

```text
Cuda compilation tools, release 12.6, V12.6.20
```

## 4. uv 虚拟环境与依赖安装

确认项目中已有虚拟环境：

```text
~/projects/CUDA-Agent/.venv
```

使用 `uv` 安装了以下依赖：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install datasets
uv pip install ninja
```

其中：

- `torch`：用于执行和编译 CUDA 扩展
- `datasets`：用于下载和读取 CUDA-Agent-Ops-6K 数据集
- `ninja`：PyTorch 编译 CUDA 扩展时使用的构建工具

## 5. PyTorch CUDA 可用性验证

执行 PyTorch 验证脚本后，确认：

```text
PyTorch: 2.13.0+cu126
PyTorch CUDA: 12.6
CUDA 可用: True
GPU: NVIDIA A100-SXM4-80GB
计算能力: (8, 0)
```

结论：PyTorch、CUDA 12.6、A100 GPU 和系统编译工具链已正确连通，可进行 CUDA 扩展编译。

## 6. A100 适配与数据集读取进度

已完成的仓库适配：

- `runner.py` 的主 CUDA 路径已更新为 `/public/app/cuda/12.6`。
- PTX/SASS 工具的 GPU 架构已更新为 A100 的 `sm_80`。
- 本地 Arrow 数据集已验证可读：`6000` 条样本，字段为 `ops`、`data_source`、`code`。

数据集文件状态：

- `CUDA-Agent-Ops-6K/data.parquet` 仅为 Git LFS 指针文件（132 字节），不可直接读取。
- 有效数据位于：

  ```text
  datasets/BytedTsinghua-SIA___cuda-agent-ops-6_k/default/0.0.0/
  44a734c78c947bfcba5189cbfd13f57a6d29a698/cuda-agent-ops-6_k-train.arrow
  ```

- 计算节点无法解析 `huggingface.co`；`runner.py` 应优先使用上述本地 Arrow 文件，避免调用 Hugging Face Hub。

## 7. 学校 Chat Completions API 尝试

尝试将原本的 `codex exec` 调用替换为学校提供的 OpenAI 兼容 Chat Completions API：

- API 通过环境变量配置，密钥不写入仓库：
  `CUDA_AGENT_API_KEY`、`CUDA_AGENT_BASE_URL`、`CUDA_AGENT_MODEL`。
- 增加 `llm_agent.py`，用于将 `SKILL.md`、`model.py`、已有 kernel 与报错发给模型，并要求模型仅返回受限 JSON 文件变更。
- `runner.py` 的生成与错误反馈阶段已改为调用 `run_agent(prompt, WORKDIR)`。
- 该方案的限制：普通 Chat Completions API 不能直接读写服务器文件或自行编译；模型代码质量也不足以稳定完成 CUDA kernel 生成与修复。因此后续计划恢复使用 Codex CLI。

## 8. 编译问题与清理策略

发现每次新任务开始前，旧任务的 `agent_workdir/kernels/*.cu`、`*.cpp` 没有被清理；而编译脚本会编译 kernels 目录下的所有 C++/CUDA 源文件，可能导致旧 kernel 与当前任务冲突。

建议在 `runner.py` 中、调用 `prepare_model()` 之前执行 `reset_generated_workspace()`：

- 删除 `agent_workdir/kernels/` 直接目录下的 `.cu`、`.cpp`、`.ptx`、`.cubin`、`.sass`；
- 删除当前生成的 `model_new.py` 与 `cuda_extension.so`；
- 保留 `model.py`、`binding.cpp`、`binding_registry.h`、`utils/` 和 `results/` 历史结果。

一次手动编译的实际结果：

- 当前仅编译了预期的 3 个文件：`binding.cpp`、`bn_digamma_max_add.cu`、`bn_digamma_max_add_binding.cpp`，说明该次没有旧 kernel 残留。
- 失败原因是生成的 `bn_digamma_max_add.cu` 使用了 `CUDART_INF_F`，但未包含 `<math_constants.h>`。
- 修复方法是在 `#include <cuda_runtime.h>` 后补充：

  ```cpp
  #include <math_constants.h>
  ```

- 编译日志仍显示 `/usr/local/cuda/bin/nvcc`，后续需要继续核对 `agent_workdir/utils/compile.py`，确保它实际使用 `/public/app/cuda/12.6/bin/nvcc`。

## 9. Codex CLI 安装进度

为保留 Codex 的本地文件编辑、命令执行和编译修复能力，决定在学校平台安装 Codex CLI，而非继续使用学校 Chat Completions API。

已确认：

- 当前节点：`anode19`。
- 家目录：`/home/scc/pb24111609`。
- Node.js：`v22.22.3`。
- npm：`10.9.8`。
- 节点可解析 `chatgpt.com`。

计划使用无 sudo、可控安装目录的 npm 安装方式：

```text
Codex 包与依赖：~/opt/openai-codex-cli/
命令快捷方式：~/.local/bin/codex
认证与配置：~/.codex/
```

待执行：

```bash
export CODEX_INSTALL_ROOT="$HOME/opt/openai-codex-cli"
export CODEX_BIN_DIR="$HOME/.local/bin"
mkdir -p "$CODEX_INSTALL_ROOT" "$CODEX_BIN_DIR"
npm install --prefix "$CODEX_INSTALL_ROOT" @openai/codex
```

安装并登录成功后，需要将 `runner.py` 中的两处 `run_agent(prompt, WORKDIR)` 恢复为 `codex exec` 调用。

## 10. 当前完成状态
## 10. 最新端到端运行结果

在 A100 节点 `anode19` 的 `agent_workdir` 中，已完成一次完整的手动编译、正确性验证与性能测试。

当前任务生成的文件：

```text
kernels/fused_activation.cu
kernels/fused_activation_binding.cpp
cuda_extension.so
```

执行结果：

```text
bash utils/compile.sh
Compile success: cuda_extension.so
[TIME] Compilation took 18.53s

python3 -m utils.verification
[PASS] check 1/5
[PASS] check 2/5
[PASS] check 3/5
[PASS] check 4/5
[PASS] check 5/5
[PASS] verify success
```

性能测试结果：

| 实现 | 耗时 |
| --- | ---: |
| Torch Baseline | 9061.079 µs |
| `torch.compile` | 1195.183 µs |
| CUDA Extension | 1465.688 µs |

结论：

- A100 上的 CUDA 扩展可以成功编译、加载、执行，并通过全部 5 项正确性检查。
- 自定义 CUDA Extension 相比原始 PyTorch baseline 明显更快。
- CUDA Extension 仍比 `torch.compile` 慢约 22.6%，尚未达到项目“至少比 `torch.compile` 快 5%”的性能目标；后续需要继续优化 `fused_activation` kernel。
- profiling 输出中的 `USDT ... profiler_start/profiler_stop` 是 profiler 的日志，不影响测量结果。

## 11. 当前完成状态

已完成：

- A100 节点与 GPU 硬件确认
- 本地 Arrow 数据集验证完成，具备离线读取条件
- 识别并定位 CUDA 源文件遗漏 `math_constants.h` 的编译错误
- 确认 Codex CLI 的 npm 安装前置条件
- A100 上的 CUDA Extension 编译成功
- A100 上的正确性验证 5/5 通过
- A100 上的性能 profiling 已成功完成

待完成：

- 安装并登录 Codex CLI
- 将 `runner.py` 从学校 API 调用恢复为 `codex exec`
- 在每个新任务开始前清理上一次生成的 kernel 文件与产物
- 确保 `compile.py` 实际使用 `/public/app/cuda/12.6/bin/nvcc`
- 在 A100 上重新编译 `cuda_extension.so`
- 执行 `utils.verification` 与 `utils.profiling` 验证正确性和性能
- 使用 Codex 继续优化 `fused_activation`，使 CUDA Extension 至少比 `torch.compile` 快 5%



source ~/.config/cuda-agent/secrets.env
module load cuda/12.6

export CUDA_HOME=/public/app/cuda/12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4

uv run python runner.py --task-id 2