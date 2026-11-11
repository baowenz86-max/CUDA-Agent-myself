# CUDA Agent Workflow Report

- Task ID: `814`
- Status: **SUCCESS**
- Operators: `["nn.Upsample", "nn.ConstantPad2d"]`
- Started: `2026-11-11T15:46:53`
- Finished: `2026-11-11T15:53:23`
- Duration: `390.65s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 48.46s |  |
| generation #1 | PASS | 71.34s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 144.01s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 39.74s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 72.83s | best; correctness_success |
| profiling | PASS | 12.27s |  |
| artifacts | PASS | 1.71s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 814 --task-end 814
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
