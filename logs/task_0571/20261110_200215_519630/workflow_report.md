# CUDA Agent Workflow Report

- Task ID: `571`
- Status: **SUCCESS**
- Operators: `["torch.isnan", "nn.ReLU6"]`
- Started: `2026-11-10T20:02:15`
- Finished: `2026-11-10T20:05:44`
- Duration: `208.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.91s |  |
| generation #1 | PASS | 35.90s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 81.21s | best; initial_candidate |
| profiling | PASS | 65.00s |  |
| artifacts | PASS | 12.55s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 571 --task-end 571
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
