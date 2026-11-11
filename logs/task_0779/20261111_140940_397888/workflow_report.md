# CUDA Agent Workflow Report

- Task ID: `779`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.prod"]`
- Started: `2026-11-11T14:09:40`
- Finished: `2026-11-11T14:16:17`
- Duration: `396.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.46s |  |
| generation #1 | PASS | 150.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 152.83s | best; initial_candidate |
| profiling | PASS | 55.97s |  |
| artifacts | PASS | 3.45s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 779 --task-end 779
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
