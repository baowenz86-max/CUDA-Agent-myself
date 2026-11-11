# CUDA Agent Workflow Report

- Task ID: `364`
- Status: **SUCCESS**
- Operators: `["torch.relu", "torch.cumsum"]`
- Started: `2026-11-09T15:08:13`
- Finished: `2026-11-09T15:13:35`
- Duration: `322.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 66.59s |  |
| generation #1 | PASS | 61.18s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 62.12s | best; initial_candidate |
| profiling | PASS | 39.56s |  |
| artifacts | PASS | 92.93s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 364 --task-end 364
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
