<<<<<<< HEAD
# CUDA-Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.24286)
[![Project Page](https://img.shields.io/badge/Blog-3858bf?style=for-the-badge&logo=homepage&logoColor=white)](https://cuda-agent.github.io/)
[![Dataset: CUDA-Agent-Ops-6K](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
## 1. Project Overview

CUDA-Agent is the first known RL-trained model to surpass advanced models such as Claude Opus-4.6 and Gemini 3 Pro on high-performance CUDA kernel generation. It achieves state-of-the-art results on KernelBench, consistently outperforming the torch.compile baseline across difficulty levels, with especially strong gains on the hardest cases. To support the LLM-based CUDA generation community, we have released our training data, expert-designed SKILL.md and agent environment.


![Benchmark Chart](./assets/benchmark_chart.png)

## 2. Dataset Release: CUDA-Agent-Ops-6K

We released the training dataset **CUDA-Agent-Ops-6K**:

- Dataset URL: [BytedTsinghua-SIA/CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
- Scale: 6,000 training samples
- Construction pipeline:
  - Collect reference operators from `torch` and `transformers`
  - Use an LLM to compose multiple operators into fused tasks
  - Apply rule-based filtering to keep executable, deterministic, and non-trivial samples
- Filtering criteria:
  - Must execute correctly in both eager mode and `torch.compile`
  - Remove stochastic operators and degenerate outputs
  - Control runtime range and remove samples highly similar to KernelBench tests to reduce contamination risk

![Data Synthesis Pipeline](./assets/data_pipeline.png)

## 3. `agent_workdir` Overview

`agent_workdir` is a standardized agent workspace example for the full loop:
implement CUDA kernels -> compile -> verify correctness -> profile performance -> iterate.

Key files in this directory:

- `SKILL.md`: workflow constraints and optimization rules for agent execution
- `model.py`: original PyTorch baseline model
- `model_new.py`: optimized model using the custom CUDA extension
- `binding.cpp` / `binding_registry.h`: shared Python binding registration infrastructure
- `kernels/`: custom CUDA/C++ kernels and their bindings
- `utils/compile.py` + `utils/compile.sh`: extension build scripts
- `utils/verification.py`: correctness validation script
- `utils/profiling.py`: performance comparison against baseline and `torch.compile`


Common commands (run inside `agent_workdir`):

```bash
bash utils/compile.sh
python3 -m utils.verification
python3 -m utils.profiling
```
![Agent Loop](./assets/agent_loop.png)
=======
# CUDA-Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.24286)
[![Project Page](https://img.shields.io/badge/Blog-3858bf?style=for-the-badge&logo=homepage&logoColor=white)](https://cuda-agent.github.io/)
[![Dataset: CUDA-Agent-Ops-6K](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
## 1. Project Overview

CUDA-Agent is the first known RL-trained model to surpass advanced models such as Claude Opus-4.6 and Gemini 3 Pro on high-performance CUDA kernel generation. It achieves state-of-the-art results on KernelBench, consistently outperforming the torch.compile baseline across difficulty levels, with especially strong gains on the hardest cases. To support the LLM-based CUDA generation community, we have released our training data, expert-designed SKILL.md and agent environment.


![Benchmark Chart](./assets/benchmark_chart.png)

## 2. Dataset Release: CUDA-Agent-Ops-6K

We released the training dataset **CUDA-Agent-Ops-6K**:

- Dataset URL: [BytedTsinghua-SIA/CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
- Scale: 6,000 training samples
- Construction pipeline:
  - Collect reference operators from `torch` and `transformers`
  - Use an LLM to compose multiple operators into fused tasks
  - Apply rule-based filtering to keep executable, deterministic, and non-trivial samples
- Filtering criteria:
  - Must execute correctly in both eager mode and `torch.compile`
  - Remove stochastic operators and degenerate outputs
  - Control runtime range and remove samples highly similar to KernelBench tests to reduce contamination risk

![Data Synthesis Pipeline](./assets/data_pipeline.png)

## 3. `agent_workdir` Overview

`agent_workdir` is a standardized agent workspace example for the full loop:
implement CUDA kernels -> compile -> verify correctness -> profile performance -> iterate.

Key files in this directory:

- `SKILL.md`: workflow constraints and optimization rules for agent execution
- `model.py`: original PyTorch baseline model
- `model_new.py`: optimized model using the custom CUDA extension
- `binding.cpp` / `binding_registry.h`: shared Python binding registration infrastructure
- `kernels/`: custom CUDA/C++ kernels and their bindings
- `utils/compile.py` + `utils/compile.sh`: extension build scripts
- `utils/verification.py`: correctness validation script
- `utils/profiling.py`: performance comparison against baseline and `torch.compile`


Common commands (run inside `agent_workdir`):

```bash
bash utils/compile.sh
python3 -m utils.verification
python3 -m utils.profiling
```
![Agent Loop](./assets/agent_loop.png)

## 4. Running the Agent

Configure the OpenAI-compatible endpoint, then run a dataset task from the
repository root:

```bash
export CUDA_AGENT_API_KEY=...
export CUDA_AGENT_BASE_URL=https://example.com/v1
export CUDA_AGENT_MODEL=your-model
python main.py --task-id 0
```

After `uv sync`, the same workflow is available as an installed command:

```bash
uv run cuda-agent --task-id 0
uv run cuda-agent-download
```

The runner performs generation, compilation with repair feedback, correctness
verification, profiling, and result archival in that order. Successful snapshots
are written to `results/task_NNNN/`. Toolchain defaults can be overridden with
`CUDA_HOME`, `CC`, `CXX`, and `CUDA_ARCH`; the defaults target CUDA 12.6, GCC 13,
and `sm_89`.

Each run writes a detailed log and a Markdown summary under
`logs/task_NNNN/<timestamp>/`. Use `--verbose` to also show DEBUG messages in the
terminal, or `--log-dir` to choose another log root. The API key is never logged.
See [docs/WORKFLOW.md](./docs/WORKFLOW.md) for the architecture and reporting
design. Copy `.env.example` as a reference when configuring the school API; the
script reads environment variables directly and does not load `.env` automatically.

Repository layout:

```text
cuda_agent/          Core package and CUDA inspection tools
scripts/             Dataset and repository maintenance commands
tests/               Isolated tests that do not modify agent_workdir
docs/                Workflow and design documentation
agent_workdir/       Runtime workspace consumed by the workflow
main.py              Backward-compatible command-line entry point
```
<<<<<<< HEAD
>>>>>>> 441d92a (Add redesigned CUDA agent project)
=======
>>>>>>> b8d3180 (Update implementation with new approach)
