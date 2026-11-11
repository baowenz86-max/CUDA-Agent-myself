# CUDA Agent Workflow Report

- Task ID: `647`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.unfold"]`
- Started: `2026-11-11T01:20:27`
- Finished: `2026-11-11T01:26:00`
- Duration: `332.49s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.44s |  |
| generation #1 | PASS | 53.03s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 164.91s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 379321 / 786432 (48.2%)
Greatest absolute difference: 9.397871971130371 at index (23, 18432, 0) (up to 0.01 allowed)
Mismatched elements: 379357 / 786432 (48.2%)
Greatest absolute difference: 11.32658576965332 at index (31, 4151, 0) (up to 0.01 allowed)
Mismatched elements: 380407 / 786432 (48.4%)
Greatest absolute difference: 9.84675407409668 at index (23, 3719, 0) (up to 0.01 allowed)
Mismatched elements: 378803 / 786432 (48.2%)
Greatest absolute difference: 9.490143775939941 at index (3, 10103, 0) (up to 0.01 allowed)
Mismatched elements: 379292 / 786432 (48.2%)
Greatest absolute difference: 9.418981552124023 at index (16, 5919, 0) (up to 0.01 allowed) |
| repair API call #1 | PASS | 29.93s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 50.62s | best; correctness_success |
| profiling | PASS | 16.84s |  |
| artifacts | PASS | 7.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 647 --task-end 647
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
