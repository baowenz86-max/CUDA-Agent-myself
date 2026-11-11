# CUDA Agent Workflow Report

- Task ID: `389`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm2d", "nn.Softmax"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:43:24`
- Duration: `484.27s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 46.04s |  |
| generation #1 | PASS | 102.77s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 95.89s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 771267 / 81920000 (0.9%)
Greatest absolute difference: 0.2693481147289276 at index (6, 386, 86, 44) (up to 0.01 allowed)
Mismatched elements: 770570 / 81920000 (0.9%)
Greatest absolute difference: 0.26199328899383545 at index (10, 152, 33, 6) (up to 0.01 allowed)
Mismatched elements: 771636 / 81920000 (0.9%)
Greatest absolute difference: 0.22702226042747498 at index (6, 471, 95, 73) (up to 0.01 allowed)
Mismatched elements: 771075 / 81920000 (0.9%)
Greatest absolute difference: 0.26262858510017395 at index (11, 22, 34, 60) (up to 0.01 allowed)
Mismatched elements: 770060 / 81920000 (0.9%)
Greatest absolute difference: 0.24953559041023254 at index (14, 291, 3, 64) (up to 0.01 allowed) |
| repair API call #1 | PASS | 44.93s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 141.75s | best; correctness_success |
| profiling | PASS | 50.02s |  |
| artifacts | PASS | 1.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 389 --task-end 389
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
