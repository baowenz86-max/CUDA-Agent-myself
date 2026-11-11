# CUDA Agent Workflow Report

- Task ID: `718`
- Status: **FAILED**
- Operators: `["torch.sum", "torch.fmod"]`
- Started: `2026-11-11T12:41:31`
- Finished: `2026-11-11T12:49:23`
- Duration: `472.87s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.96s |  |
| generation #1 | PASS | 29.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 76.96s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0718/build/forced_compile/cuda_extension.so: undefined symbol: custom_sum_dim1_launcher
[TIME] Compilation took 27.29s |
| repair API call #1 | PASS | 10.01s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 75.52s | best; pipeline_progress; torch.AcceleratorError: CUDA error: operation not supported on global/shared address space
Search for `cudaErrorInvalidAddressSpace' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
 |
| repair API call #2 | PASS | 64.17s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 99.83s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 2 / 524288 (0.0%)
Greatest absolute difference: 9.99993896484375 at index (11, 13, 7) (up to 0.01 allowed)
Mismatched elements: 1 / 524288 (0.0%)
Greatest absolute difference: 9.999954223632812 at index (56, 6, 22) (up to 0.01 allowed)
Mismatched elements: 3 / 524288 (0.0%)
Greatest absolute difference: 9.999954223632812 at index (8, 17, 5) (up to 0.01 allowed)
Greatest absolute difference: 9.999961853027344 at index (454, 30, 4) (up to 0.01 allowed)
Greatest absolute difference: 9.99993896484375 at index (240, 18, 28) (up to 0.01 allowed) |
| repair API call #3 | PASS | 14.10s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 56.35s | best; final_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 1 / 524288 (0.0%)
Greatest absolute difference: 9.999992370605469 at index (467, 30, 17) (up to 0.01 allowed)
Greatest absolute difference: 9.999992370605469 at index (111, 11, 16) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [512, 32, 32], "dtype": "torch.float32", "device": "cuda:0", "stride": [1024, 32, 1], "numel": 524288}, "expected": {"type": "tensor", "shape": [512, 32, 32], "dtype": "torch.float32", "device": "cuda:0", "stride": [1024, 32, 1], "numel": 524288}, "kind": "tensor_values", "mismatched": 1, "numel": 524288, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 1.9073486328125e-06, "max_absolute_error": 9.999992370605469, "max_relative_error": 9.999992370605468e+30, "samples": [{"index": [467, 30, 17], "actual": "-9.999992370605469", "expected": "-0.0"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 1 / 524288 (0.0%)\nGreatest absolute difference: 9.999992370605469 at index (467, 30, 17) (up to 0.01 allowed)\nGreatest relative difference: inf at index (467, 30, 17) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 718 --task-end 718
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
