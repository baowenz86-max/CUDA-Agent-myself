# CUDA Agent Workflow Report

- Task ID: `460`
- Status: **FAILED**
- Operators: `["torch.unbind", "nn.PixelShuffle"]`
- Started: `2026-11-10T00:17:56`
- Finished: `2026-11-10T00:24:57`
- Duration: `421.14s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.17s |  |
| generation #1 | PASS | 132.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.04s | best; initial_candidate; debug_forward: prohibited torch computation torch.unbind
debug_forward: prohibited torch computation torch.stack |
| repair API call #1 | PASS | 10.57s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 76.69s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 18.40s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 73.43s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #3 | PASS | 15.96s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 81.48s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"kind": "runtime", "message": "AcceleratorError: CUDA error: an illegal memory access was encountered\nSearch for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"}

## Reproduction

```bash
python main.py --task-start 460 --task-end 460
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
