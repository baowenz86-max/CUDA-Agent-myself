# CUDA Agent Workflow Report

- Task ID: `560`
- Status: **SUCCESS**
- Operators: `["F.softplus", "nn.AvgPool2d"]`
- Started: `2026-11-10T19:40:32`
- Finished: `2026-11-10T19:43:54`
- Duration: `201.89s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.28s |  |
| generation #1 | PASS | 64.85s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 108.38s | best; initial_candidate |
| profiling | PASS | 24.25s |  |
| artifacts | PASS | 1.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 560 --task-end 560
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
