# CUDA Agent Workflow Report

- Task ID: `408`
- Status: **SUCCESS**
- Operators: `["torch.max", "torch.prod"]`
- Started: `2026-11-09T22:19:13`
- Finished: `2026-11-09T22:25:21`
- Duration: `367.97s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 36.16s |  |
| generation #1 | PASS | 83.92s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 25.42s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0408/build/forced_compile/cuda_extension.so: undefined symbol: reduce_prod_4d_launcher
[TIME] Compilation took 22.78s |
| repair API call #1 | PASS | 24.56s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 81.53s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 38.15s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 42.62s | best; correctness_success |
| profiling | PASS | 26.00s |  |
| artifacts | PASS | 8.00s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 408 --task-end 408
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
