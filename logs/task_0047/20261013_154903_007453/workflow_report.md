# CUDA Agent Workflow Report

- Task ID: `47`
- Status: **FAILED**
- Operators: `["nn.KLDivLoss", "F.l1_loss", "nn.ParameterList"]`
- Started: `2026-10-13T15:49:03`
- Finished: `2026-10-13T16:05:18`
- Duration: `975.24s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.07s |  |
| generation #1 | PASS | 217.65s | DeepSeek API |
| compile candidate #1 | FAIL | 20.17s | best |
| interface candidate #2 | FAIL | 26.68s | best |
| interface candidate #3 | FAIL | 23.39s | rejected |
| generation #2 | PASS | 232.60s | DeepSeek API |
| sources candidate #4 | FAIL | 0.04s | rejected |
| interface candidate #5 | FAIL | 21.31s | rejected |
| interface candidate #6 | FAIL | 19.45s | rejected |
| generation #3 | PASS | 224.07s | DeepSeek API |
| compile candidate #7 | FAIL | 12.71s | rejected |
| interface candidate #8 | FAIL | 20.86s | rejected |
| interface candidate #9 | FAIL | 33.00s | rejected |

## Failure Summary

Best candidate 2, stage interface: Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 5.57s

Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0047/utils/preflight.py", line 58, in main
    raise ValueError('\n'.join(errors))
ValueError: custom_linear: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0047/20261013_154903_007453/correctness/interface.json


## Reproduction

```bash
python main.py --task-start 47 --task-end 47
```
