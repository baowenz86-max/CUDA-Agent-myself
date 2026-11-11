# CUDA Agent Workflow Report

- Task ID: `367`
- Status: **SUCCESS**
- Operators: `["torch.sigmoid", "F.adaptive_avg_pool2d"]`
- Started: `2026-11-09T15:08:13`
- Finished: `2026-11-09T15:13:34`
- Duration: `321.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 59.58s |  |
| generation #1 | PASS | 49.54s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 51.40s | best; initial_candidate |
| profiling | PASS | 43.84s |  |
| artifacts | PASS | 117.03s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 367 --task-end 367
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
