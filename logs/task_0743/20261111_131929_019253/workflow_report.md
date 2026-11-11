# CUDA Agent Workflow Report

- Task ID: `743`
- Status: **SUCCESS**
- Operators: `["F.softmax", "nn.PixelShuffle"]`
- Started: `2026-11-11T13:19:29`
- Finished: `2026-11-11T13:26:34`
- Duration: `425.23s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.39s |  |
| generation #1 | PASS | 65.80s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 89.24s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 87151926 / 100663296 (86.6%)
Greatest absolute difference: 0.920096218585968 at index (113, 2, 115, 31) (up to 0.01 allowed)
Mismatched elements: 87158981 / 100663296 (86.6%)
Greatest absolute difference: 0.9227762222290039 at index (414, 0, 184, 56) (up to 0.01 allowed)
Mismatched elements: 87168652 / 100663296 (86.6%)
Greatest absolute difference: 0.9304079413414001 at index (176, 2, 134, 83) (up to 0.01 allowed)
Mismatched elements: 87165780 / 100663296 (86.6%)
Greatest absolute difference: 0.9210799336433411 at index (46, 1, 186, 7) (up to 0.01 allowed)
Mismatched elements: 87151729 / 100663296 (86.6%)
Greatest absolute difference: 0.9353330135345459 at index (133, 1, 108, 212) (up to 0.01 allowed) |
| repair API call #1 | PASS | 41.43s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 65.15s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 86.51s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 27.98s | best; correctness_success |
| profiling | PASS | 11.78s |  |
| artifacts | PASS | 1.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 743 --task-end 743
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
