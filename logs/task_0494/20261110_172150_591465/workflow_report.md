# CUDA Agent Workflow Report

- Task ID: `494`
- Status: **SUCCESS**
- Operators: `["F.softplus", "F.leaky_relu"]`
- Started: `2026-11-10T17:21:50`
- Finished: `2026-11-10T17:24:32`
- Duration: `162.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.64s |  |
| generation #1 | PASS | 45.27s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 74.57s | best; initial_candidate |
| profiling | PASS | 12.26s |  |
| artifacts | PASS | 1.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 494 --task-end 494
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
