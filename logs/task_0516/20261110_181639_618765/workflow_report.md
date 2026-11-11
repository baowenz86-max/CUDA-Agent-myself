# CUDA Agent Workflow Report

- Task ID: `516`
- Status: **SUCCESS**
- Operators: `["F.group_norm"]`
- Started: `2026-11-10T18:16:39`
- Finished: `2026-11-10T18:19:16`
- Duration: `157.30s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.12s |  |
| generation #1 | PASS | 57.72s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 41.27s | best; initial_candidate |
| profiling | PASS | 11.51s |  |
| artifacts | PASS | 1.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 516 --task-end 516
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
