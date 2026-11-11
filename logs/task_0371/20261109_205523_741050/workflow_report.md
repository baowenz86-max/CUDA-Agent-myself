# CUDA Agent Workflow Report

- Task ID: `371`
- Status: **SUCCESS**
- Operators: `["nn.CrossMapLRN2d"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:03:47`
- Duration: `503.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 42.71s |  |
| generation #1 | PASS | 56.79s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 332.37s | best; initial_candidate |
| profiling | PASS | 70.31s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 371 --task-end 371
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
