# CUDA Agent Workflow Report

- Task ID: `289`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool3d"]`
- Started: `2026-11-07T14:37:21`
- Finished: `2026-11-07T14:43:52`
- Duration: `390.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.94s |  |
| generation #1 | PASS | 124.50s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 129.55s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0289/kernels/custom_ops.cu(83): error: kernel launch from __device__ or __global__ functions requires separate compilation mode
      max_pool3d_kernel<<<256, 256>>>(
      ^
 |
| repair API call #1 | PASS | 13.25s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 62.43s | best |
| profiling | PASS | 39.53s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 289 --task-end 289
```
