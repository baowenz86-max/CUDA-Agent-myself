# CUDA Agent Workflow Report

- Task ID: `826`
- Status: **FAILED**
- Operators: `["torch.clone", "torch.rot90"]`
- Started: `2026-11-11T17:49:13`
- Finished: `2026-11-11T17:57:57`
- Duration: `523.97s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 96.62s |  |
| generation #1 | PASS | 36.44s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 55.30s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 155897467 / 157286400 (99.1%)
Greatest absolute difference: 8.39208698272705 at index (226, 301, 2, 257) (up to 0.01 allowed)
Mismatched elements: 155898452 / 157286400 (99.1%)
Greatest absolute difference: 8.756637573242188 at index (268, 267, 1, 113) (up to 0.01 allowed)
Mismatched elements: 155898670 / 157286400 (99.1%)
Greatest absolute difference: 8.080221176147461 at index (494, 156, 1, 204) (up to 0.01 allowed)
Mismatched elements: 155898892 / 157286400 (99.1%)
Greatest absolute difference: 8.075374603271484 at index (195, 286, 1, 230) (up to 0.01 allowed)
Mismatched elements: 155894368 / 157286400 (99.1%)
Greatest absolute difference: 8.284778594970703 at index (499, 139, 2, 194) (up to 0.01 allowed) |
| repair API call #1 | PASS | 14.42s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 79.08s | best; final_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 155411999 / 157286400 (98.8%)
Greatest absolute difference: 7.882905960083008 at index (264, 37, 1, 72) (up to 0.01 allowed)
Mismatched elements: 155413799 / 157286400 (98.8%)
Greatest absolute difference: 8.170502662658691 at index (115, 11, 2, 167) (up to 0.01 allowed)
Mismatched elements: 155411151 / 157286400 (98.8%)
Greatest absolute difference: 7.904510498046875 at index (423, 130, 0, 257) (up to 0.01 allowed)
Mismatched elements: 155409950 / 157286400 (98.8%)
Greatest absolute difference: 8.236740112304688 at index (213, 68, 0, 202) (up to 0.01 allowed)
Mismatched elements: 155409824 / 157286400 (98.8%)
Greatest absolute difference: 8.3899564743042 at index (143, 207, 1, 249) (up to 0.01 allowed) |
| repair API call #2 | PASS | 23.12s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 40.57s | rejected; final_mismatch_regression; AssertionError: Tensor-likes are not close!

Mismatched elements: 155896562 / 157286400 (99.1%)
Greatest absolute difference: 8.587669372558594 at index (325, 14, 0, 186) (up to 0.01 allowed)
Mismatched elements: 155897371 / 157286400 (99.1%)
Greatest absolute difference: 8.53532600402832 at index (508, 5, 0, 1) (up to 0.01 allowed)
Mismatched elements: 155898508 / 157286400 (99.1%)
Greatest absolute difference: 8.972391128540039 at index (280, 107, 2, 242) (up to 0.01 allowed)
Mismatched elements: 155897390 / 157286400 (99.1%)
Greatest absolute difference: 8.279982566833496 at index (238, 290, 2, 175) (up to 0.01 allowed)
Mismatched elements: 155897675 / 157286400 (99.1%)
Greatest absolute difference: 8.762239456176758 at index (17, 306, 2, 245) (up to 0.01 allowed) |
| repair API call #3 | PASS | 139.83s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 36.37s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 155411999 / 157286400 (98.8%)
Greatest absolute difference: 7.882905960083008 at index (264, 37, 1, 72) (up to 0.01 allowed)
Mismatched elements: 155413799 / 157286400 (98.8%)
Greatest absolute difference: 8.170502662658691 at index (115, 11, 2, 167) (up to 0.01 allowed)
Mismatched elements: 155411151 / 157286400 (98.8%)
Greatest absolute difference: 7.904510498046875 at index (423, 130, 0, 257) (up to 0.01 allowed)
Mismatched elements: 155409950 / 157286400 (98.8%)
Greatest absolute difference: 8.236740112304688 at index (213, 68, 0, 202) (up to 0.01 allowed)
Mismatched elements: 155409824 / 157286400 (98.8%)
Greatest absolute difference: 8.3899564743042 at index (143, 207, 1, 249) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [512, 320, 3, 320], "dtype": "torch.float32", "device": "cuda:0", "stride": [307200, 960, 320, 1], "numel": 157286400}, "expected": {"type": "tensor", "shape": [512, 320, 3, 320], "dtype": "torch.float32", "device": "cuda:0", "stride": [307200, 320, 102400, 1], "numel": 157286400}, "kind": "tensor_values", "mismatched": 155411999, "numel": 157286400, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9880828793843587, "max_absolute_error": 7.882905721664429, "max_relative_error": 2.8895022869110103e+30, "samples": [{"index": [0, 0, 0, 1], "actual": "0.695512592792511", "expected": "-0.6331536173820496"}, {"index": [0, 0, 0, 2], "actual": "-1.5107221603393555", "expected": "0.0971783921122551"}, {"index": [0, 0, 0, 3], "actual": "-1.1835440397262573", "expected": "0.7584304213523865"}, {"index": [0, 0, 0, 4], "actual": "0.4574941396713257", "expected": "0.2579244375228882"}, {"index": [0, 0, 0, 5], "actual": "-0.32857462763786316", "expected": "0.8258455991744995"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 155411999 / 157286400 (98.8%)\nGreatest absolute difference: 7.882905960083008 at index (264, 37, 1, 72) (up to 0.01 allowed)\nGreatest relative difference: inf at index (62, 0, 2, 231) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 826 --task-end 826
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
