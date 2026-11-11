# CUDA Agent Workflow Report

- Task ID: `361`
- Status: **SUCCESS**
- Operators: `["nn.Hardswish"]`
- Started: `2026-11-09T14:59:56`
- Finished: `2026-11-09T15:05:27`
- Duration: `331.12s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 78.69s |  |
| generation #1 | PASS | 36.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 184.30s | best; initial_candidate |
| profiling | PASS | 30.43s |  |
| artifacts | PASS | 1.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 361 --task-end 361
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
