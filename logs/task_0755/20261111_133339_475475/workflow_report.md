# CUDA Agent Workflow Report

- Task ID: `755`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.gt"]`
- Started: `2026-11-11T13:33:39`
- Finished: `2026-11-11T13:37:33`
- Duration: `234.12s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.39s |  |
| generation #1 | PASS | 48.28s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 141.30s | best; initial_candidate |
| profiling | PASS | 24.38s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 755 --task-end 755
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
