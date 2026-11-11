# CUDA Agent Workflow Report

- Task ID: `713`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.cos"]`
- Started: `2026-11-11T11:32:58`
- Finished: `2026-11-11T11:36:07`
- Duration: `188.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.60s |  |
| generation #1 | PASS | 41.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 101.17s | best; initial_candidate |
| profiling | PASS | 26.23s |  |
| artifacts | PASS | 1.77s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 713 --task-end 713
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
