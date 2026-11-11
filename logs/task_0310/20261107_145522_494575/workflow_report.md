# CUDA Agent Workflow Report

- Task ID: `310`
- Status: **FAILED**
- Operators: `["nn.InstanceNorm2d", "torch.selu"]`
- Started: `2026-11-07T14:55:22`
- Finished: `2026-11-07T15:14:31`
- Duration: `1148.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.39s |  |
| generation #1 | PASS | 52.14s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 670.50s | best; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 25.30s

[PASSED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0310/20261107_145522_494575/correctness/interface.json


Command timed out; the scheduling lease has been released. |
| repair API call #1 | FAIL | 79.12s | Code repairs used: 0/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #2 | FAIL | 15.25s | Code repairs used: 0/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #3 | FAIL | 328.43s | Code repairs used: 0/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |

## Failure Summary

Repair API failed 3 consecutive times; code repairs used 0/3. Best candidate 1, stage verification: Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 25.30s

[PASSED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0310/20261107_145522_494575/correctness/interface.json


Command timed out; the scheduling lease has been released.

## Reproduction

```bash
python main.py --task-start 310 --task-end 310
```
