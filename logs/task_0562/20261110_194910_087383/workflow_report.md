# CUDA Agent Workflow Report

- Task ID: `562`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.round"]`
- Started: `2026-11-10T19:49:10`
- Finished: `2026-11-10T19:54:34`
- Duration: `324.78s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 43.18s |  |
| generation #1 | PASS | 58.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 136.10s | best; initial_candidate |
| profiling | PASS | 85.45s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 562 --task-end 562
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
