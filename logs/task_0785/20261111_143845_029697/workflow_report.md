# CUDA Agent Workflow Report

- Task ID: `785`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.gelu"]`
- Started: `2026-11-11T14:38:45`
- Finished: `2026-11-11T14:42:33`
- Duration: `228.34s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.73s |  |
| generation #1 | PASS | 44.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 84.61s | best; initial_candidate |
| profiling | PASS | 62.28s |  |
| artifacts | PASS | 1.77s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 785 --task-end 785
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
