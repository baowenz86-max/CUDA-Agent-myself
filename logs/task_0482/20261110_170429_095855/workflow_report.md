# CUDA Agent Workflow Report

- Task ID: `482`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "nn.MaxPool3d"]`
- Started: `2026-11-10T17:04:29`
- Finished: `2026-11-10T17:13:14`
- Duration: `525.14s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 64.24s |  |
| generation #1 | PASS | 75.44s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 64.43s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0482/kernels/custom_ops.cu(161): error: identifier "printf" is undefined
      do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0482/kernels/custom_ops.cu", 161, cudaGetErrorString(err)); cudaDeviceReset(); } } while(0);
                                                                           ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0482/kernels/custom_ops.cu(202): error: identifier "printf" is undefined
      do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0482/kernels/custom_ops.cu", 202, cudaGetErrorString(err)); cudaDeviceReset(); } } while(0); |
| repair API call #1 | PASS | 51.59s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 194.98s | best; pipeline_progress; TypeError: maxpool3d(): incompatible function arguments. The following argument types are supported:
    1. (x: torch.Tensor, kernel_size: typing.SupportsInt \| typing.SupportsIndex, stride: typing.SupportsInt \| typing.SupportsIndex) -> torch.Tensor

Invoked with: tensor([[[[[ 5.9891e-03,  3.7982e+00, -5.7090e-01,  ..., -1.1595e+00, |
| repair API call #2 | PASS | 22.41s | Code repairs used: 2/3; base candidate #2 |
| repair #2: sources candidate #3 | FAIL | 0.08s | rejected; missing_normal_observations; maxpool3d.kernel_size: C++ type/nullable differs
maxpool3d.stride: C++ type/nullable differs |
| repair API call #3 | PASS | 10.48s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | PASS | 24.26s | best; correctness_success |
| profiling | PASS | 12.48s |  |
| artifacts | PASS | 1.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 482 --task-end 482
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
