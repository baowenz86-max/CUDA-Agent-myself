# CUDA Agent Workflow Report

- Task ID: `450`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.CrossEntropyLoss"]`
- Started: `2026-11-10T00:03:32`
- Finished: `2026-11-10T00:12:09`
- Duration: `516.78s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.55s |  |
| generation #1 | PASS | 82.72s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 77.88s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 2583576 / 2621440 (98.6%)
Greatest absolute difference: 2.6116583347320557 at index (5, 7058, 6) (up to 0.01 allowed)
Mismatched elements: 2583502 / 2621440 (98.6%)
Greatest absolute difference: 2.5723683834075928 at index (8, 5712, 6) (up to 0.01 allowed)
Mismatched elements: 2583614 / 2621440 (98.6%)
Greatest absolute difference: 2.5986857414245605 at index (8, 2027, 6) (up to 0.01 allowed)
Mismatched elements: 2583578 / 2621440 (98.6%)
Greatest absolute difference: 2.5895869731903076 at index (18, 5563, 6) (up to 0.01 allowed)
Mismatched elements: 2583497 / 2621440 (98.6%)
Greatest absolute difference: 2.552396774291992 at index (21, 5252, 6) (up to 0.01 allowed) |
| repair API call #1 | PASS | 48.23s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 37.87s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 2583576 / 2621440 (98.6%)
Greatest absolute difference: 2.6116583347320557 at index (5, 7058, 6) (up to 0.01 allowed)
Mismatched elements: 2583502 / 2621440 (98.6%)
Greatest absolute difference: 2.5723683834075928 at index (8, 5712, 6) (up to 0.01 allowed)
Mismatched elements: 2583614 / 2621440 (98.6%)
Greatest absolute difference: 2.5986857414245605 at index (8, 2027, 6) (up to 0.01 allowed)
Mismatched elements: 2583578 / 2621440 (98.6%)
Greatest absolute difference: 2.5895869731903076 at index (18, 5563, 6) (up to 0.01 allowed)
Mismatched elements: 2583497 / 2621440 (98.6%)
Greatest absolute difference: 2.552396774291992 at index (21, 5252, 6) (up to 0.01 allowed) |
| repair API call #2 | PASS | 54.56s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 28.53s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 2583576 / 2621440 (98.6%)
Greatest absolute difference: 2.6116583347320557 at index (5, 7058, 6) (up to 0.01 allowed)
Mismatched elements: 2583502 / 2621440 (98.6%)
Greatest absolute difference: 2.5723683834075928 at index (8, 5712, 6) (up to 0.01 allowed)
Mismatched elements: 2583614 / 2621440 (98.6%)
Greatest absolute difference: 2.5986857414245605 at index (8, 2027, 6) (up to 0.01 allowed)
Mismatched elements: 2583578 / 2621440 (98.6%)
Greatest absolute difference: 2.5895869731903076 at index (18, 5563, 6) (up to 0.01 allowed)
Mismatched elements: 2583497 / 2621440 (98.6%)
Greatest absolute difference: 2.552396774291992 at index (21, 5252, 6) (up to 0.01 allowed) |
| repair API call #3 | PASS | 120.04s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | PASS | 25.99s | best; correctness_success |
| profiling | PASS | 10.98s |  |
| artifacts | PASS | 1.17s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 450 --task-end 450
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
