# CUDA Agent Workflow Report

- Task ID: `610`
- Status: **SUCCESS**
- Operators: `["torch.equal", "nn.BatchNorm1d"]`
- Started: `2026-11-11T00:00:34`
- Finished: `2026-11-11T00:05:01`
- Duration: `266.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.12s |  |
| generation #1 | PASS | 85.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.06s | best; initial_candidate; forward: prohibited torch computation torch.equal
debug_forward: prohibited torch computation torch.equal |
| repair API call #1 | PASS | 110.83s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 44.02s | best; correctness_success |
| profiling | PASS | 18.26s |  |
| artifacts | PASS | 1.56s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 610 --task-end 610
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
