# CUDA Agent Workflow Report

- Task ID: `722`
- Status: **FAILED**
- Operators: `["torch.mean", "torch.gt"]`
- Started: `2026-11-11T12:49:25`
- Finished: `2026-11-11T12:55:43`
- Duration: `377.71s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.57s |  |
| generation #1 | PASS | 39.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 119.54s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 22.03s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 42.64s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 37.51s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 36.42s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 201326592 (0.0%)
Greatest absolute difference: 1 at index (12, 1, 1683, 115) (up to 0.01 allowed)
Mismatched elements: 2 / 201326592 (0.0%)
Greatest absolute difference: 1 at index (11, 2, 747, 1396) (up to 0.01 allowed)
Greatest absolute difference: 1 at index (3, 0, 1044, 1225) (up to 0.01 allowed) |
| repair API call #3 | PASS | 17.26s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 28.24s | best; final_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 201326592 (0.0%)
Greatest absolute difference: 1 at index (11, 2, 747, 1396) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [16, 3, 2048, 2048], "dtype": "torch.bool", "device": "cuda:0", "stride": [12582912, 4194304, 2048, 1], "numel": 201326592}, "expected": {"type": "tensor", "shape": [16, 3, 2048, 2048], "dtype": "torch.bool", "device": "cuda:0", "stride": [12582912, 4194304, 2048, 1], "numel": 201326592}, "kind": "tensor_values", "mismatched": 1, "numel": 201326592, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 4.967053731282552e-09, "max_absolute_error": 1.0, "max_relative_error": 9.999999999999999e+29, "samples": [{"index": [11, 2, 747, 1396], "actual": "True", "expected": "False"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 1 / 201326592 (0.0%)\nGreatest absolute difference: 1 at index (11, 2, 747, 1396) (up to 0.01 allowed)\nGreatest relative difference: inf at index (11, 2, 747, 1396) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 722 --task-end 722
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
