# CUDA Agent Workflow Report

- Task ID: `364`
- Status: **SUCCESS**
- Operators: `["torch.relu", "torch.cumsum"]`
- Started: `2026-11-09T14:59:56`
- Finished: `2026-11-09T15:04:48`
- Duration: `291.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 51.65s |  |
| generation #1 | PASS | 45.80s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 116.06s | best; initial_candidate |
| profiling | PASS | 44.77s |  |
| artifacts | PASS | 33.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 364 --task-end 364
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
