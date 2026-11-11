# CUDA Agent Workflow Report

- Task ID: `612`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.expm1"]`
- Started: `2026-11-11T00:07:06`
- Finished: `2026-11-11T00:15:05`
- Duration: `479.33s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.18s |  |
| generation #1 | PASS | 63.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 182.68s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 162.63s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 48.53s | best; correctness_success |
| profiling | PASS | 13.40s |  |
| artifacts | PASS | 1.03s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 612 --task-end 612
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
