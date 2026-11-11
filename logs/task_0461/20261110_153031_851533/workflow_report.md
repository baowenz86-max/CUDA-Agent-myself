# CUDA Agent Workflow Report

- Task ID: `461`
- Status: **SUCCESS**
- Operators: `["torch.max", "nn.ZeroPad2d"]`
- Started: `2026-11-10T15:30:31`
- Finished: `2026-11-10T15:35:47`
- Duration: `316.07s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 30.61s |  |
| generation #1 | PASS | 88.30s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 163.18s | best; initial_candidate |
| profiling | PASS | 32.15s |  |
| artifacts | PASS | 1.79s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 461 --task-end 461
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
