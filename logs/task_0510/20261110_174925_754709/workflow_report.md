# CUDA Agent Workflow Report

- Task ID: `510`
- Status: **SUCCESS**
- Operators: `["nn.AvgPool3d", "nn.Tanh"]`
- Started: `2026-11-10T17:49:25`
- Finished: `2026-11-10T17:53:04`
- Duration: `218.41s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.08s |  |
| generation #1 | PASS | 48.59s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 49.58s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 197663 / 31850496 (0.6%)
Greatest absolute difference: 0.11202651262283325 at index (219, 1, 0, 40) (up to 0.01 allowed)
Mismatched elements: 197634 / 31850496 (0.6%)
Greatest absolute difference: 0.11202645301818848 at index (23, 0, 0, 8) (up to 0.01 allowed)
Mismatched elements: 197662 / 31850496 (0.6%)
Greatest absolute difference: 0.11202651262283325 at index (63, 2, 0, 16) (up to 0.01 allowed)
Mismatched elements: 197592 / 31850496 (0.6%)
Greatest absolute difference: 0.11202645301818848 at index (6, 0, 0, 15) (up to 0.01 allowed)
Mismatched elements: 197485 / 31850496 (0.6%)
Greatest absolute difference: 0.11202645301818848 at index (20, 2, 0, 84) (up to 0.01 allowed) |
| repair API call #1 | PASS | 39.24s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 44.35s | best; correctness_success |
| profiling | PASS | 13.37s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 510 --task-end 510
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
