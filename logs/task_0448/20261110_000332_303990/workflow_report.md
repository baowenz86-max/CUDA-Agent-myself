# CUDA Agent Workflow Report

- Task ID: `448`
- Status: **SUCCESS**
- Operators: `["torch.clone"]`
- Started: `2026-11-10T00:03:32`
- Finished: `2026-11-10T00:06:31`
- Duration: `179.39s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.30s |  |
| generation #1 | PASS | 38.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 53.19s | best; initial_candidate |
| profiling | PASS | 41.89s |  |
| artifacts | PASS | 20.74s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 448 --task-end 448
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
