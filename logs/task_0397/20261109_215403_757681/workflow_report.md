# CUDA Agent Workflow Report

- Task ID: `397`
- Status: **SUCCESS**
- Operators: `["nn.Softshrink"]`
- Started: `2026-11-09T21:54:03`
- Finished: `2026-11-09T21:59:23`
- Duration: `319.51s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.40s |  |
| generation #1 | PASS | 41.00s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 105.82s | best; initial_candidate |
| profiling | PASS | 42.76s |  |
| artifacts | PASS | 94.46s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 397 --task-end 397
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
