# CUDA Agent Workflow Report

- Task ID: `668`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "div"]`
- Started: `2026-11-11T10:27:13`
- Finished: `2026-11-11T10:30:08`
- Duration: `174.71s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.20s |  |
| generation #1 | PASS | 41.64s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 47.80s | best; initial_candidate |
| profiling | PASS | 24.83s |  |
| artifacts | PASS | 29.15s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 668 --task-end 668
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
