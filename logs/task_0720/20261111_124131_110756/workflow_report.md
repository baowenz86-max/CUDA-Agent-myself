# CUDA Agent Workflow Report

- Task ID: `720`
- Status: **SUCCESS**
- Operators: `["F.linear"]`
- Started: `2026-11-11T12:41:31`
- Finished: `2026-11-11T12:44:45`
- Duration: `194.86s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.47s |  |
| generation #1 | PASS | 40.54s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 85.28s | best; initial_candidate |
| profiling | PASS | 17.55s |  |
| artifacts | PASS | 27.90s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 720 --task-end 720
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
