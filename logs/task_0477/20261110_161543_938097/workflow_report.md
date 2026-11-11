# CUDA Agent Workflow Report

- Task ID: `477`
- Status: **SUCCESS**
- Operators: `["nn.InstanceNorm3d"]`
- Started: `2026-11-10T16:15:43`
- Finished: `2026-11-10T16:19:03`
- Duration: `199.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.41s |  |
| generation #1 | PASS | 68.90s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 79.28s | best; initial_candidate |
| profiling | PASS | 35.41s |  |
| artifacts | PASS | 1.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 477 --task-end 477
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
