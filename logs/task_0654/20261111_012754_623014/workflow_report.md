# CUDA Agent Workflow Report

- Task ID: `654`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.argmin"]`
- Started: `2026-11-11T01:27:54`
- Finished: `2026-11-11T01:31:52`
- Duration: `237.48s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.24s |  |
| generation #1 | PASS | 35.90s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 115.12s | best; initial_candidate |
| profiling | PASS | 55.28s |  |
| artifacts | PASS | 1.86s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 654 --task-end 654
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
