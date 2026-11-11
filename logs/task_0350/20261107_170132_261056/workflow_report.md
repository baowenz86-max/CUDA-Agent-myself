# CUDA Agent Workflow Report

- Task ID: `350`
- Status: **FAILED**
- Operators: `["nn.ConvTranspose1d", "nn.Sequential"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:08:47`
- Duration: `434.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 30.26s |  |
| generation #1 | PASS | 78.64s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 113.22s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0350/kernels/custom_ops_binding.cpp:6:10: fatal error: cudnn.h: No such file or directory
    6 \| #include <cudnn.h>
      \|          ^~~~~~~~~
compilation terminated. |
| repair API call #1 | PASS | 40.78s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 82.63s | best; ValueError: conv_transpose1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'stride: typing.SupportsInt \| typing.SupportsIndex = 2', 'padding: typing.SupportsInt \| typing.SupportsIndex = 1', 'output_padding: typing.SupportsInt \| typing.SupportsIndex = 1']]
conv1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'stride: typing.SupportsInt \| typing.SupportsIndex = 1', 'padding: typing.SupportsInt \| typing.SupportsIndex = 1']]
max_pool1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'kernel_size: typing.SupportsInt \| typing.SupportsIndex = 2', 'stride: typing.SupportsInt \| typing.SupportsIndex = 2', 'padding: typing.SupportsInt \| typing.SupportsIndex = 0']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0350/20261107_170132_261056/correctness/interface.json |
| repair API call #2 | PASS | 28.68s | Code repairs used: 2/3; base candidate #2 |
| repair #2: compile candidate #3 | FAIL | 21.29s | rejected; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compilation failed.
initialization failed
[TIME] Compilation took 21.21s
 |
| repair API call #3 | PASS | 23.53s | Code repairs used: 3/3; base candidate #2 |
| repair #3: compile candidate #4 | FAIL | 15.76s | rejected; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compilation failed.
initialization failed
[TIME] Compilation took 15.68s
 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage interface: Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 25.28s

Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0350/utils/preflight.py", line 58, in main
    raise ValueError('\n'.join(errors))
ValueError: conv_transpose1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'stride: typing.SupportsInt | typing.SupportsIndex = 2', 'padding: typing.SupportsInt | typing.SupportsIndex = 1', 'output_padding: typing.SupportsInt | typing.SupportsIndex = 1']]
conv1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor', 'stride: typing.SupportsInt | typing.SupportsIndex = 1', 'padding: typing.SupportsInt | typing.SupportsIndex = 1']]
max_pool1d_forward: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'kernel_size: typing.SupportsInt | typing.SupportsIndex = 2', 'stride: typing.SupportsInt | typing.SupportsIndex = 2', 'padding: typing.SupportsInt | typing.SupportsIndex = 0']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0350/20261107_170132_261056/correctness/interface.json


## Reproduction

```bash
python main.py --task-start 350 --task-end 350
```
