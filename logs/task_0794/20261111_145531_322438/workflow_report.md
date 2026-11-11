# CUDA Agent Workflow Report

- Task ID: `794`
- Status: **SUCCESS**
- Operators: `["torch.logical_or", "F.mse_loss"]`
- Started: `2026-11-11T14:55:31`
- Finished: `2026-11-11T14:59:40`
- Duration: `248.71s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.94s |  |
| generation #1 | PASS | 67.41s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 163.39s | best; initial_candidate |
| profiling | PASS | 11.88s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 794 --task-end 794
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
