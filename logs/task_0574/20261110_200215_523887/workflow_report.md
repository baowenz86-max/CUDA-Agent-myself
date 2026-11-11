# CUDA Agent Workflow Report

- Task ID: `574`
- Status: **SUCCESS**
- Operators: `["torch.isnan", "nn.BatchNorm1d"]`
- Started: `2026-11-10T20:02:15`
- Finished: `2026-11-10T20:06:52`
- Duration: `276.90s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.49s |  |
| generation #1 | PASS | 72.47s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 159.15s | best; initial_candidate |
| profiling | PASS | 25.92s |  |
| artifacts | PASS | 1.81s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 574 --task-end 574
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
