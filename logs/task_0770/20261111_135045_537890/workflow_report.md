# CUDA Agent Workflow Report

- Task ID: `770`
- Status: **SUCCESS**
- Operators: `["torch.max", "torch.any"]`
- Started: `2026-11-11T13:50:45`
- Finished: `2026-11-11T13:57:04`
- Duration: `378.60s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.68s |  |
| generation #1 | PASS | 94.15s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 137.29s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 65.20s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 43.74s | best; correctness_success |
| profiling | PASS | 14.22s |  |
| artifacts | PASS | 1.27s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 770 --task-end 770
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
