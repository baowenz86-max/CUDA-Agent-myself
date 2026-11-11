# CUDA Agent Workflow Report

- Task ID: `418`
- Status: **SUCCESS**
- Operators: `["F.lp_pool2d"]`
- Started: `2026-11-09T23:23:34`
- Finished: `2026-11-09T23:27:07`
- Duration: `212.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.31s |  |
| generation #1 | PASS | 56.15s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 67.09s | best; initial_candidate |
| profiling | PASS | 22.24s |  |
| artifacts | PASS | 47.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 418 --task-end 418
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
