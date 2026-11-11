# CUDA Agent Workflow Report

- Task ID: `820`
- Status: **FAILED**
- Operators: `["torch.ceil", "nn.Upsample"]`
- Started: `2026-11-11T15:46:53`
- Finished: `2026-11-11T15:56:36`
- Duration: `583.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 60.12s |  |
| generation #1 | PASS | 72.73s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 228.97s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 54.76s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 35.89s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 3 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (413, 1, 79, 66) (up to 0.01 allowed)
Mismatched elements: 4 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (149, 1, 32, 187) (up to 0.01 allowed)
Mismatched elements: 2 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (44, 1, 72, 160) (up to 0.01 allowed)
Mismatched elements: 1 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (171, 1, 154, 134) (up to 0.01 allowed)
Greatest absolute difference: 1.0 at index (58, 2, 6, 228) (up to 0.01 allowed) |
| repair API call #2 | PASS | 45.34s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 39.29s | best; final_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 2 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (413, 1, 79, 66) (up to 0.01 allowed)
Mismatched elements: 3 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (149, 1, 32, 187) (up to 0.01 allowed)
Mismatched elements: 1 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (44, 1, 72, 160) (up to 0.01 allowed)
Greatest absolute difference: 1.0 at index (58, 2, 6, 228) (up to 0.01 allowed) |
| repair API call #3 | PASS | 30.03s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 15.65s | rejected; normal_observation_regression; AssertionError: Tensor-likes are not close!

Mismatched elements: 3 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (413, 1, 79, 66) (up to 0.01 allowed)
Mismatched elements: 4 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (149, 1, 32, 187) (up to 0.01 allowed)
Mismatched elements: 2 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (44, 1, 72, 160) (up to 0.01 allowed)
Mismatched elements: 1 / 100663296 (0.0%)
Greatest absolute difference: 1.0 at index (171, 1, 154, 134) (up to 0.01 allowed)
Greatest absolute difference: 1.0 at index (58, 2, 6, 228) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [512, 3, 256, 256], "dtype": "torch.float32", "device": "cuda:0", "stride": [196608, 65536, 256, 1], "numel": 100663296}, "expected": {"type": "tensor", "shape": [512, 3, 256, 256], "dtype": "torch.float32", "device": "cuda:0", "stride": [196608, 65536, 256, 1], "numel": 100663296}, "kind": "tensor_values", "mismatched": 2, "numel": 100663296, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 1.9868214925130207e-08, "max_absolute_error": 1.0, "max_relative_error": 0.5, "samples": [{"index": [413, 1, 79, 66], "actual": "1.0", "expected": "2.0"}, {"index": [506, 2, 166, 30], "actual": "1.0", "expected": "2.0"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 2 / 100663296 (0.0%)\nGreatest absolute difference: 1.0 at index (413, 1, 79, 66) (up to 0.01 allowed)\nGreatest relative difference: 0.5 at index (413, 1, 79, 66) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 820 --task-end 820
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
