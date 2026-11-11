# CUDA Agent Workflow Report

- Task ID: `691`
- Status: **SUCCESS**
- Operators: `["torch.sum", "F.mse_loss"]`
- Started: `2026-11-11T11:00:43`
- Finished: `2026-11-11T11:05:26`
- Duration: `283.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 43.52s |  |
| generation #1 | PASS | 60.09s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 64.30s | best; initial_candidate; AssertionError: Scalars are not close!

Expected 2.000237226486206 but got 536934592.0.
Absolute difference: 536934589.9997628 (up to 0.01 allowed)
Expected 1.9996507167816162 but got 536777152.0.
Absolute difference: 536777150.0003493 (up to 0.01 allowed)
Expected 2.000265121459961 but got 536942016.0.
Absolute difference: 536942013.9997349 (up to 0.01 allowed)
Expected 2.000127077102661 but got 536904960.0.
Absolute difference: 536904957.9998729 (up to 0.01 allowed)
Expected 1.9998407363891602 but got 536828160.0.
Absolute difference: 536828158.00015926 (up to 0.01 allowed) |
| repair API call #1 | PASS | 24.06s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 66.35s | best; correctness_success |
| profiling | PASS | 20.35s |  |
| artifacts | PASS | 2.15s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 691 --task-end 691
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
