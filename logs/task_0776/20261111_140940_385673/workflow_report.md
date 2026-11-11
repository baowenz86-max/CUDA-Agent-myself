# CUDA Agent Workflow Report

- Task ID: `776`
- Status: **FAILED**
- Operators: `["torch.stack", "nn.AdaptiveAvgPool1d"]`
- Started: `2026-11-11T14:09:40`
- Finished: `2026-11-11T14:18:56`
- Duration: `556.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.29s |  |
| generation #1 | PASS | 56.47s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 91.47s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 62925558 / 67108864 (93.8%)
Greatest absolute difference: 1.0585851669311523 at index (38, 4, 463, 10) (up to 0.01 allowed)
Mismatched elements: 62930773 / 67108864 (93.8%)
Greatest absolute difference: 0.9367929697036743 at index (0, 6, 551, 11) (up to 0.01 allowed)
Mismatched elements: 62932903 / 67108864 (93.8%)
Greatest absolute difference: 1.0927571058273315 at index (35, 9, 899, 60) (up to 0.01 allowed)
Mismatched elements: 62932170 / 67108864 (93.8%)
Greatest absolute difference: 1.0201611518859863 at index (44, 13, 383, 54) (up to 0.01 allowed)
Mismatched elements: 62928398 / 67108864 (93.8%)
Greatest absolute difference: 1.0233690738677979 at index (0, 2, 424, 42) (up to 0.01 allowed) |
| repair API call #1 | PASS | 36.03s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 207.58s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 30.57s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 25.76s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #3 | PASS | 38.98s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 57.60s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"actual": {"type": "tensor", "shape": [64, 16, 1024, 64], "dtype": "torch.float32", "device": "cuda:0", "stride": [1048576, 65536, 64, 1], "numel": 67108864}, "expected": {"type": "tensor", "shape": [64, 16, 1024, 64], "dtype": "torch.float32", "device": "cuda:0", "stride": [1048576, 65536, 64, 1], "numel": 67108864}, "kind": "tensor_values", "mismatched": 62925558, "numel": 67108864, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9376638829708099, "max_absolute_error": 1.0585851967334747, "max_relative_error": 26228497.0, "samples": [{"index": [0, 0, 0, 1], "actual": "-0.008438185788691044", "expected": "-0.05715533718466759"}, {"index": [0, 0, 0, 2], "actual": "-0.11240621656179428", "expected": "-0.008438185788691044"}, {"index": [0, 0, 0, 3], "actual": "0.07264572381973267", "expected": "-0.07311846315860748"}, {"index": [0, 0, 0, 4], "actual": "-0.05444115027785301", "expected": "-0.11240621656179428"}, {"index": [0, 0, 0, 5], "actual": "-0.015811240300536156", "expected": "0.08111780136823654"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 62925558 / 67108864 (93.8%)\nGreatest absolute difference: 1.0585851669311523 at index (38, 4, 463, 10) (up to 0.01 allowed)\nGreatest relative difference: 26228496.0 at index (9, 9, 971, 22) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 776 --task-end 776
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
