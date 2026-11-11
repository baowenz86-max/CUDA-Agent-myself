# CUDA Agent Workflow Report

- Task ID: `526`
- Status: **SUCCESS**
- Operators: `["sub"]`
- Started: `2026-11-10T18:29:21`
- Finished: `2026-11-10T18:31:17`
- Duration: `115.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.41s |  |
| generation #1 | PASS | 35.01s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 38.80s | best; initial_candidate |
| profiling | PASS | 28.46s |  |
| artifacts | PASS | 2.73s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 526 --task-end 526
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
