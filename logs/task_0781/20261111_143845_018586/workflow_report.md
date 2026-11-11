# CUDA Agent Workflow Report

- Task ID: `781`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.pad"]`
- Started: `2026-11-11T14:38:45`
- Finished: `2026-11-11T14:43:08`
- Duration: `263.18s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.46s |  |
| generation #1 | PASS | 60.39s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 156.12s | best; initial_candidate |
| profiling | PASS | 15.38s |  |
| artifacts | PASS | 1.66s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 781 --task-end 781
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
