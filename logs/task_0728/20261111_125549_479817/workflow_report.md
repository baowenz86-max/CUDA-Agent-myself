# CUDA Agent Workflow Report

- Task ID: `728`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.squeeze"]`
- Started: `2026-11-11T12:55:49`
- Finished: `2026-11-11T13:01:06`
- Duration: `316.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.79s |  |
| generation #1 | PASS | 45.64s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 171.69s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 266274797 / 536870912 (49.6%)
Greatest absolute difference: 5.767325401306152 at index (389, 11, 87, 232) (up to 0.01 allowed)
Mismatched elements: 265839525 / 536870912 (49.5%)
Greatest absolute difference: 6.767683506011963 at index (360, 15, 29, 125) (up to 0.01 allowed)
Mismatched elements: 267203427 / 536870912 (49.8%)
Greatest absolute difference: inf at index (256, 11, 150, 192) (up to 0.01 allowed)
Mismatched elements: 267498407 / 536870912 (49.8%)
Mismatched elements: 267807478 / 536870912 (49.9%) |
| repair API call #1 | PASS | 12.16s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 59.63s | best; correctness_success |
| profiling | PASS | 15.89s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 728 --task-end 728
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
