# CUDA Agent Workflow Report

- Task ID: `761`
- Status: **SUCCESS**
- Operators: `["F.softmax", "torch.squeeze"]`
- Started: `2026-11-11T13:43:50`
- Finished: `2026-11-11T13:48:37`
- Duration: `287.94s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.52s |  |
| generation #1 | PASS | 39.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 24.84s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0761/kernels/custom_ops.cu(79): error: calling a __device__ function("T1  ::ceil_div<unsigned long> (T1, T1)") from a __host__ function("squeeze_launcher") is not allowed
      int blocks = (int)ceil_div<size_t>(numel, threads);
                        ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0761/kernels/custom_ops.cu(86): error: calling a __device__ function("_Z8ceil_divI1?E1?1?1?") from a __host__ function("add_scalar_launcher") is not allowed |
| repair API call #1 | PASS | 12.07s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 117.44s | best; correctness_success |
| profiling | PASS | 84.81s |  |
| artifacts | PASS | 2.44s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 761 --task-end 761
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
