# CUDA Agent Workflow Report

- Task ID: `630`
- Status: **FAILED**
- Operators: `["torch.abs", "torch.einsum"]`
- Started: `2026-11-11T00:24:17`
- Finished: `2026-11-11T00:34:19`
- Duration: `602.03s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.95s |  |
| generation #1 | PASS | 96.87s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; custom_ops_binding.cpp: prohibited torch computation in generated C++
forward: cuda_extension.custom_abs has no visible pybind registration
forward: cuda_extension.custom_reduce_lastdim has no visible pybind registration
debug_forward: cuda_extension.custom_abs has no visible pybind registration
debug_forward: cuda_extension.custom_reduce_lastdim has no visible pybind registration |
| repair API call #1 | PASS | 51.08s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 60.54s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 654154 / 786432 (83.2%)
Greatest absolute difference: 90.98977661132812 at index (419, 2, 120) (up to 0.01 allowed)
Mismatched elements: 654235 / 786432 (83.2%)
Greatest absolute difference: 98.74859619140625 at index (255, 2, 108) (up to 0.01 allowed)
Mismatched elements: 653537 / 786432 (83.1%)
Greatest absolute difference: 91.14666748046875 at index (5, 0, 350) (up to 0.01 allowed)
Mismatched elements: 654323 / 786432 (83.2%)
Greatest absolute difference: 94.37835693359375 at index (77, 0, 59) (up to 0.01 allowed)
Mismatched elements: 654437 / 786432 (83.2%)
Greatest absolute difference: 91.29095458984375 at index (276, 2, 387) (up to 0.01 allowed) |
| repair API call #2 | FAIL | 251.04s | Code repairs used: 1/3; base candidate #2; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #3 | PASS | 21.47s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 43.80s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 654154 / 786432 (83.2%)
Greatest absolute difference: 90.98977661132812 at index (419, 2, 120) (up to 0.01 allowed)
Mismatched elements: 654235 / 786432 (83.2%)
Greatest absolute difference: 98.74859619140625 at index (255, 2, 108) (up to 0.01 allowed)
Mismatched elements: 653537 / 786432 (83.1%)
Greatest absolute difference: 91.14666748046875 at index (5, 0, 350) (up to 0.01 allowed)
Mismatched elements: 654323 / 786432 (83.2%)
Greatest absolute difference: 94.37835693359375 at index (77, 0, 59) (up to 0.01 allowed)
Mismatched elements: 654437 / 786432 (83.2%)
Greatest absolute difference: 91.29095458984375 at index (276, 2, 387) (up to 0.01 allowed) |
| repair API call #4 | PASS | 20.18s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 42.13s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 654154 / 786432 (83.2%)
Greatest absolute difference: 90.98977661132812 at index (419, 2, 120) (up to 0.01 allowed)
Mismatched elements: 654235 / 786432 (83.2%)
Greatest absolute difference: 98.74859619140625 at index (255, 2, 108) (up to 0.01 allowed)
Mismatched elements: 653537 / 786432 (83.1%)
Greatest absolute difference: 91.14666748046875 at index (5, 0, 350) (up to 0.01 allowed)
Mismatched elements: 654323 / 786432 (83.2%)
Greatest absolute difference: 94.37835693359375 at index (77, 0, 59) (up to 0.01 allowed)
Mismatched elements: 654437 / 786432 (83.2%)
Greatest absolute difference: 91.29095458984375 at index (276, 2, 387) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [512, 3, 512], "dtype": "torch.float32", "device": "cuda:0", "stride": [1536, 512, 1], "numel": 786432}, "expected": {"type": "tensor", "shape": [512, 3, 512], "dtype": "torch.float32", "device": "cuda:0", "stride": [1536, 512, 1], "numel": 786432}, "kind": "tensor_values", "mismatched": 654154, "numel": 786432, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.8317998250325521, "max_absolute_error": 90.98977661132812, "max_relative_error": 0.2473736052450403, "samples": [{"index": [0, 0, 0], "actual": "437.20330810546875", "expected": "406.7288818359375"}, {"index": [0, 0, 1], "actual": "409.8708190917969", "expected": "414.4435119628906"}, {"index": [0, 0, 2], "actual": "426.4389953613281", "expected": "392.856201171875"}, {"index": [0, 0, 3], "actual": "413.16339111328125", "expected": "394.5604553222656"}, {"index": [0, 0, 4], "actual": "411.40234375", "expected": "387.65753173828125"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 654154 / 786432 (83.2%)\nGreatest absolute difference: 90.98977661132812 at index (419, 2, 120) (up to 0.01 allowed)\nGreatest relative difference: 0.24737361073493958 at index (419, 2, 120) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 630 --task-end 630
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
