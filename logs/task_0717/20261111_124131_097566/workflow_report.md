# CUDA Agent Workflow Report

- Task ID: `717`
- Status: **SUCCESS**
- Operators: `["torch.cat", "F.smooth_l1_loss"]`
- Started: `2026-11-11T12:41:31`
- Finished: `2026-11-11T12:44:51`
- Duration: `200.21s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.64s |  |
| generation #1 | PASS | 43.14s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 48.14s | best; initial_candidate |
| profiling | PASS | 21.71s |  |
| artifacts | PASS | 67.46s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 717 --task-end 717
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
