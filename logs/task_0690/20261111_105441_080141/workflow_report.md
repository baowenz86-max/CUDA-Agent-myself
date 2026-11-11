# CUDA Agent Workflow Report

- Task ID: `690`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.cos"]`
- Started: `2026-11-11T10:54:41`
- Finished: `2026-11-11T10:59:30`
- Duration: `289.60s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.09s |  |
| generation #1 | PASS | 42.59s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 116.42s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 502003732 / 536870912 (93.5%)
Greatest absolute difference: 2.0 at index (99, 4, 58, 61) (up to 0.01 allowed)
Mismatched elements: 502007654 / 536870912 (93.5%)
Greatest absolute difference: 2.0 at index (109, 3, 73, 82) (up to 0.01 allowed)
Mismatched elements: 502010470 / 536870912 (93.5%)
Greatest absolute difference: 2.0 at index (215, 11, 192, 199) (up to 0.01 allowed)
Mismatched elements: 502006410 / 536870912 (93.5%)
Greatest absolute difference: 2.0 at index (132, 3, 124, 159) (up to 0.01 allowed)
Mismatched elements: 502003901 / 536870912 (93.5%)
Greatest absolute difference: 2.0 at index (293, 3, 65, 144) (up to 0.01 allowed) |
| repair API call #1 | PASS | 23.97s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 59.07s | best; correctness_success |
| profiling | PASS | 22.62s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 690 --task-end 690
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
