# CUDA Agent Workflow Report

- Task ID: `158`
- Status: **FAILED**
- Operators: `["torch.mean", "add"]`
- Started: `2026-11-06T17:29:04`
- Finished: `2026-11-06T17:44:30`
- Duration: `925.50s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 38.19s |  |
| generation #1 | PASS | 60.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 134.77s | best; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 91.68s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 143.51s | rejected; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 150.67s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 71.60s | rejected; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #3 | PASS | 189.28s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 39.90s | rejected; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"kind": "runtime", "message": "AcceleratorError: CUDA error: an illegal memory access was encountered\nSearch for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"}

## Reproduction

```bash
python main.py --task-start 158 --task-end 158
```
