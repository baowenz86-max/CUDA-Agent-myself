# CUDA Agent Workflow Report

- Task ID: `40`
- Status: **FAILED**
- Operators: `["nn.ConvTranspose1d", "torch.roll", "F.linear"]`
- Started: `2026-10-13T15:09:05`
- Finished: `2026-10-13T15:15:28`
- Duration: `382.50s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.78s |  |
| generation #1 | PASS | 164.90s | DeepSeek API |
| compile candidate #1 | FAIL | 59.05s | best |
| interface candidate #2 | FAIL | 35.45s | best |
| interface candidate #3 | FAIL | 32.29s | rejected |
| generation #2 | FAIL | 0.51s | DeepSeek API |
| generation #3 | FAIL | 0.51s | DeepSeek API |

## Failure Summary

Best candidate 2, stage interface: Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 5.17s

Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0040/utils/preflight.py", line 58, in main
    raise ValueError('\n'.join(errors))
ValueError: conv_transpose1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'stride: typing.SupportsInt | typing.SupportsIndex = 2', 'padding: typing.SupportsInt | typing.SupportsIndex = 1', 'output_padding: typing.SupportsInt | typing.SupportsIndex = 1']]
roll_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'shifts: typing.SupportsInt | typing.SupportsIndex = 1', 'dims: typing.SupportsInt | typing.SupportsIndex = 2']]
linear_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0040/20261013_150905_808741/correctness/interface.json


## Reproduction

```bash
python main.py --task-start 40 --task-end 40
```
