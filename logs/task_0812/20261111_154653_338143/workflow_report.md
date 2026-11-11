# CUDA Agent Workflow Report

- Task ID: `812`
- Status: **FAILED**
- Operators: `["nn.Upsample", "torch.rot90"]`
- Started: `2026-11-11T15:46:53`
- Finished: `2026-11-11T16:06:16`
- Duration: `1163.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.98s |  |
| generation #1 | PASS | 100.82s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 219.52s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 268.55s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 19.32s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 28.94s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 18.55s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 265162161 / 268435456 (98.8%)
Greatest absolute difference: 5.431822299957275 at index (12, 0, 0, 1023) (up to 0.01 allowed)
Mismatched elements: 265165998 / 268435456 (98.8%)
Greatest absolute difference: 5.795709133148193 at index (13, 4, 0, 678) (up to 0.01 allowed)
Mismatched elements: 265163902 / 268435456 (98.8%)
Greatest absolute difference: 5.462888717651367 at index (9, 9, 0, 0) (up to 0.01 allowed)
Mismatched elements: 265161618 / 268435456 (98.8%)
Greatest absolute difference: 5.617099285125732 at index (9, 20, 0, 699) (up to 0.01 allowed)
Mismatched elements: 265161357 / 268435456 (98.8%)
Greatest absolute difference: 5.684572696685791 at index (12, 45, 0, 139) (up to 0.01 allowed) |
| repair API call #3 | PASS | 439.91s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 20.72s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [16, 64, 256, 1024], "dtype": "torch.float32", "device": "cuda:0", "stride": [16777216, 262144, 1024, 1], "numel": 268435456}, "expected": {"type": "tensor", "shape": [16, 64, 256, 1024], "dtype": "torch.float32", "device": "cuda:0", "stride": [16777216, 262144, 1, 256], "numel": 268435456}, "kind": "tensor_values", "mismatched": 265162161, "numel": 268435456, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9878060258924961, "max_absolute_error": 5.431822419166565, "max_relative_error": 8.379914164543151e+29, "samples": [{"index": [0, 0, 0, 0], "actual": "0.43173548579216003", "expected": "-2.3804376125335693"}, {"index": [0, 0, 0, 1], "actual": "-0.1605532169342041", "expected": "-2.1870665550231934"}, {"index": [0, 0, 0, 2], "actual": "-1.3451305627822876", "expected": "-1.8003246784210205"}, {"index": [0, 0, 0, 3], "actual": "-1.2859243154525757", "expected": "-1.1358680725097656"}, {"index": [0, 0, 0, 4], "actual": "0.01706555485725403", "expected": "-0.19369709491729736"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 265162161 / 268435456 (98.8%)\nGreatest absolute difference: 5.431822299957275 at index (12, 0, 0, 1023) (up to 0.01 allowed)\nGreatest relative difference: inf at index (3, 8, 60, 841) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 812 --task-end 812
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
