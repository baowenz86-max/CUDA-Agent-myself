# CUDA Agent Workflow Report

- Task ID: `620`
- Status: **SUCCESS**
- Operators: `["F.hardtanh", "nn.AvgPool3d"]`
- Started: `2026-11-11T00:15:10`
- Finished: `2026-11-11T00:18:30`
- Duration: `200.83s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.35s |  |
| generation #1 | PASS | 69.73s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 80.91s | best; initial_candidate |
| profiling | PASS | 36.66s |  |
| artifacts | PASS | 1.13s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 620 --task-end 620
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
