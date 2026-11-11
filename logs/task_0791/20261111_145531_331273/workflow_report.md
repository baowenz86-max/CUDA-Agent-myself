# CUDA Agent Workflow Report

- Task ID: `791`
- Status: **SUCCESS**
- Operators: `["torch.logical_or", "torch.round"]`
- Started: `2026-11-11T14:55:31`
- Finished: `2026-11-11T14:58:46`
- Duration: `195.54s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.03s |  |
| generation #1 | PASS | 44.96s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 104.50s | best; initial_candidate |
| profiling | PASS | 32.50s |  |
| artifacts | PASS | 1.51s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 791 --task-end 791
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
