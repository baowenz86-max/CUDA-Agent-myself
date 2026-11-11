# CUDA Agent Workflow Report

- Task ID: `521`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "F.smooth_l1_loss"]`
- Started: `2026-11-10T18:19:58`
- Finished: `2026-11-10T18:23:30`
- Duration: `211.93s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 37.02s |  |
| generation #1 | PASS | 92.34s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 64.79s | best; initial_candidate |
| profiling | PASS | 14.47s |  |
| artifacts | PASS | 3.26s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 521 --task-end 521
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
