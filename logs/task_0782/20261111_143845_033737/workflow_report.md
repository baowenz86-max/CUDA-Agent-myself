# CUDA Agent Workflow Report

- Task ID: `782`
- Status: **SUCCESS**
- Operators: `["*"]`
- Started: `2026-11-11T14:38:45`
- Finished: `2026-11-11T14:44:07`
- Duration: `322.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 42.61s |  |
| generation #1 | PASS | 49.28s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 202.17s | best; initial_candidate |
| profiling | PASS | 26.92s |  |
| artifacts | PASS | 1.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 782 --task-end 782
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
