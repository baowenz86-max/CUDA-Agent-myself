# CUDA Agent Workflow Report

- Task ID: `451`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.tan"]`
- Started: `2026-11-10T00:12:10`
- Finished: `2026-11-10T00:17:55`
- Duration: `345.19s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.04s |  |
| generation #1 | PASS | 44.92s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 152.97s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 16.68s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 82.44s | best; correctness_success |
| profiling | PASS | 25.75s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 451 --task-end 451
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
