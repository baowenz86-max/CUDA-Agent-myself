# CUDA Agent Workflow Report

- Task ID: `395`
- Status: **SUCCESS**
- Operators: `["nn.functional", "nn.AvgPool2d"]`
- Started: `2026-11-09T21:54:03`
- Finished: `2026-11-09T21:59:45`
- Duration: `341.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 55.53s |  |
| generation #1 | PASS | 71.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 167.34s | best; initial_candidate |
| profiling | PASS | 45.88s |  |
| artifacts | PASS | 1.75s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 395 --task-end 395
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
