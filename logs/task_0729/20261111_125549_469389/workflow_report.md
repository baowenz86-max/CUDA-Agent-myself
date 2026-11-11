# CUDA Agent Workflow Report

- Task ID: `729`
- Status: **FAILED**
- Operators: `["torch.sum", "torch.round"]`
- Started: `2026-11-11T12:55:49`
- Finished: `2026-11-11T13:06:03`
- Duration: `614.15s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 37.06s |  |
| generation #1 | PASS | 52.32s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 140.98s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 1536 (0.1%)
Greatest absolute difference: 1.0 at index (332, 0) (up to 0.01 allowed) |
| repair API call #1 | PASS | 26.59s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 98.86s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 1536 (0.1%)
Greatest absolute difference: 1.0 at index (332, 0) (up to 0.01 allowed) |
| repair API call #2 | PASS | 66.50s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 81.22s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 1536 (0.1%)
Greatest absolute difference: 1.0 at index (332, 0) (up to 0.01 allowed) |
| repair API call #3 | PASS | 68.95s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 30.29s | rejected; missing_normal_observations; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 16.08s

[PASSED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0729/20261111_125549_469389/correctness/interface.json

 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"actual": {"type": "tensor", "shape": [512, 3], "dtype": "torch.float32", "device": "cuda:0", "stride": [3, 1], "numel": 1536}, "expected": {"type": "tensor", "shape": [512, 3], "dtype": "torch.float32", "device": "cuda:0", "stride": [3, 1], "numel": 1536}, "kind": "tensor_values", "mismatched": 1, "numel": 1536, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.0006510416666666666, "max_absolute_error": 1.0, "max_relative_error": 0.125, "samples": [{"index": [332, 0], "actual": "-9.0", "expected": "-8.0"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 1 / 1536 (0.1%)\nGreatest absolute difference: 1.0 at index (332, 0) (up to 0.01 allowed)\nGreatest relative difference: 0.125 at index (332, 0) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 729 --task-end 729
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
