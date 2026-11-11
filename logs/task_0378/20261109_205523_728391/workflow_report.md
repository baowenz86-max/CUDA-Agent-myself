# CUDA Agent Workflow Report

- Task ID: `378`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm2d"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:02:00`
- Duration: `396.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 39.18s |  |
| generation #1 | PASS | 53.21s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 156.96s | best; initial_candidate |
| profiling | PASS | 144.69s |  |
| artifacts | PASS | 2.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 378 --task-end 378
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
