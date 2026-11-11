# CUDA Agent Workflow Report

- Task ID: `287`
- Status: **FAILED**
- Operators: `["torch.bmm", "torch.tanh"]`
- Started: `2026-11-07T14:37:21`
- Finished: `2026-11-07T14:44:41`
- Duration: `439.93s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.02s |  |
| generation #1 | PASS | 49.73s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 192.15s | best; ValueError: batch_norm_eval_forward: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'running_mean: torch.Tensor', 'running_var: torch.Tensor', 'eps: typing.SupportsFloat \| typing.SupportsIndex = 1e-05']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0287/20261107_143721_256873/correctness/interface.json |
| repair API call #1 | PASS | 19.67s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 61.28s | rejected; ValueError: batch_norm_eval_forward: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'running_mean: torch.Tensor', 'running_var: torch.Tensor', 'eps: typing.SupportsFloat \| typing.SupportsIndex = 1e-05']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0287/20261107_143721_256873/correctness/interface.json |
| repair API call #2 | PASS | 20.25s | Code repairs used: 2/3; base candidate #1 |
| repair #2: interface candidate #3 | FAIL | 32.42s | rejected; ValueError: batch_norm_eval_forward: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'running_mean: torch.Tensor', 'running_var: torch.Tensor', 'eps: typing.SupportsFloat \| typing.SupportsIndex = 1e-05']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0287/20261107_143721_256873/correctness/interface.json |
| repair API call #3 | PASS | 17.85s | Code repairs used: 3/3; base candidate #1 |
| repair #3: interface candidate #4 | FAIL | 17.35s | rejected; ValueError: batch_norm_eval_forward: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'running_mean: torch.Tensor', 'running_var: torch.Tensor', 'eps: typing.SupportsFloat \| typing.SupportsIndex = 1e-05']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0287/20261107_143721_256873/correctness/interface.json |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage interface: Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 26.93s

Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0287/utils/preflight.py", line 58, in main
    raise ValueError('\n'.join(errors))
ValueError: batch_norm_eval_forward: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'running_mean: torch.Tensor', 'running_var: torch.Tensor', 'eps: typing.SupportsFloat | typing.SupportsIndex = 1e-05']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0287/20261107_143721_256873/correctness/interface.json


## Reproduction

```bash
python main.py --task-start 287 --task-end 287
```
