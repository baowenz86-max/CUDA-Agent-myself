# CUDA Agent Workflow Report

- Task ID: `346`
- Status: **FAILED**
- Operators: `["nn.UpsamplingBilinear2d"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:09:15`
- Duration: `463.21s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.75s |  |
| generation #1 | PASS | 37.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 42.17s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 186935185 / 201326592 (92.9%)
Greatest absolute difference: 2.3856911659240723 at index (7, 2, 2025, 46) (up to 0.01 allowed)
Mismatched elements: 186938601 / 201326592 (92.9%)
Greatest absolute difference: 2.4304280281066895 at index (5, 0, 78, 92) (up to 0.01 allowed)
Mismatched elements: 186937531 / 201326592 (92.9%)
Greatest absolute difference: 2.3779220581054688 at index (0, 0, 2025, 2043) (up to 0.01 allowed)
Mismatched elements: 186946701 / 201326592 (92.9%)
Greatest absolute difference: 2.404191493988037 at index (13, 2, 44, 8) (up to 0.01 allowed)
Mismatched elements: 186932764 / 201326592 (92.9%)
Greatest absolute difference: 2.4495584964752197 at index (6, 0, 50, 132) (up to 0.01 allowed) |
| repair API call #1 | PASS | 24.90s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 241.89s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 186935184 / 201326592 (92.9%)
Greatest absolute difference: 2.3856911659240723 at index (7, 2, 2025, 46) (up to 0.01 allowed)
Mismatched elements: 186938602 / 201326592 (92.9%)
Greatest absolute difference: 2.4304280281066895 at index (5, 0, 78, 92) (up to 0.01 allowed)
Mismatched elements: 186937531 / 201326592 (92.9%)
Greatest absolute difference: 2.3779220581054688 at index (0, 0, 2025, 2043) (up to 0.01 allowed)
Mismatched elements: 186946701 / 201326592 (92.9%)
Greatest absolute difference: 2.404191493988037 at index (13, 2, 44, 8) (up to 0.01 allowed)
Mismatched elements: 186932767 / 201326592 (92.9%)
Greatest absolute difference: 2.4495584964752197 at index (6, 0, 50, 132) (up to 0.01 allowed) |
| repair API call #2 | PASS | 17.04s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 24.61s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 186935184 / 201326592 (92.9%)
Greatest absolute difference: 2.3856911659240723 at index (7, 2, 2025, 46) (up to 0.01 allowed)
Mismatched elements: 186938602 / 201326592 (92.9%)
Greatest absolute difference: 2.4304280281066895 at index (5, 0, 78, 92) (up to 0.01 allowed)
Mismatched elements: 186937531 / 201326592 (92.9%)
Greatest absolute difference: 2.3779220581054688 at index (0, 0, 2025, 2043) (up to 0.01 allowed)
Mismatched elements: 186946701 / 201326592 (92.9%)
Greatest absolute difference: 2.404191493988037 at index (13, 2, 44, 8) (up to 0.01 allowed)
Mismatched elements: 186932767 / 201326592 (92.9%)
Greatest absolute difference: 2.4495584964752197 at index (6, 0, 50, 132) (up to 0.01 allowed) |
| repair API call #3 | PASS | 42.12s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 29.45s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 186935185 / 201326592 (92.9%)
Greatest absolute difference: 2.3856911659240723 at index (7, 2, 2025, 46) (up to 0.01 allowed)
Mismatched elements: 186938601 / 201326592 (92.9%)
Greatest absolute difference: 2.4304280281066895 at index (5, 0, 78, 92) (up to 0.01 allowed)
Mismatched elements: 186937531 / 201326592 (92.9%)
Greatest absolute difference: 2.3779220581054688 at index (0, 0, 2025, 2043) (up to 0.01 allowed)
Mismatched elements: 186946701 / 201326592 (92.9%)
Greatest absolute difference: 2.404191493988037 at index (13, 2, 44, 8) (up to 0.01 allowed)
Mismatched elements: 186932764 / 201326592 (92.9%)
Greatest absolute difference: 2.4495584964752197 at index (6, 0, 50, 132) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"actual": {"type": "tensor", "shape": [16, 3, 2048, 2048], "dtype": "torch.float32", "device": "cuda:0", "stride": [12582912, 4194304, 2048, 1], "numel": 201326592}, "expected": {"type": "tensor", "shape": [16, 3, 2048, 2048], "dtype": "torch.float32", "device": "cuda:0", "stride": [12582912, 4194304, 2048, 1], "numel": 201326592}, "kind": "tensor_values", "mismatched": 186935185, "numel": 201326592, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9285171081622442, "max_absolute_error": 2.3856911659240723, "max_relative_error": 124492309.04270463, "samples": [{"index": [0, 0, 0, 1], "actual": "0.8293729424476624", "expected": "0.576486349105835"}, {"index": [0, 0, 0, 2], "actual": "0.3231051564216614", "expected": "0.07046594470739365"}, {"index": [0, 0, 0, 3], "actual": "-0.4632182717323303", "expected": "-0.9948450326919556"}, {"index": [0, 0, 0, 4], "actual": "-1.529597520828247", "expected": "-2.0607032775878906"}, {"index": [0, 0, 0, 5], "actual": "-1.4620583057403564", "expected": "-0.8642638921737671"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 186935185 / 201326592 (92.9%)\nGreatest absolute difference: 2.3856911659240723 at index (7, 2, 2025, 46) (up to 0.01 allowed)\nGreatest relative difference: 124492312.0 at index (0, 1, 231, 1101) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 346 --task-end 346
```
