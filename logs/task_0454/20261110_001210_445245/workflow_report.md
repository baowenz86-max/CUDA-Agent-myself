# CUDA Agent Workflow Report

- Task ID: `454`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.AdaptiveAvgPool1d"]`
- Started: `2026-11-10T00:12:10`
- Finished: `2026-11-10T00:14:59`
- Duration: `168.82s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 15.75s |  |
| generation #1 | PASS | 37.25s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 47.67s | best; initial_candidate |
| profiling | PASS | 21.79s |  |
| artifacts | PASS | 46.30s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 454 --task-end 454
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
