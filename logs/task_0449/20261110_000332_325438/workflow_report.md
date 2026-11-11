# CUDA Agent Workflow Report

- Task ID: `449`
- Status: **SUCCESS**
- Operators: `["F.local_response_norm"]`
- Started: `2026-11-10T00:03:32`
- Finished: `2026-11-10T00:06:32`
- Duration: `180.46s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.51s |  |
| generation #1 | PASS | 51.09s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 54.41s | best; initial_candidate |
| profiling | PASS | 39.64s |  |
| artifacts | PASS | 3.76s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 449 --task-end 449
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
