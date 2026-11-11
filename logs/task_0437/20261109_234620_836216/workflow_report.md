# CUDA Agent Workflow Report

- Task ID: `437`
- Status: **SUCCESS**
- Operators: `["nn.functional", "nn.Hardswish"]`
- Started: `2026-11-09T23:46:20`
- Finished: `2026-11-09T23:48:52`
- Duration: `151.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 41.40s |  |
| generation #1 | PASS | 37.39s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 59.51s | best; initial_candidate |
| profiling | PASS | 11.46s |  |
| artifacts | PASS | 1.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 437 --task-end 437
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
