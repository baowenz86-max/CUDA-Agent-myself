# CUDA Agent Workflow Report

- Task ID: `282`
- Status: **SUCCESS**
- Operators: `["nn.InstanceNorm3d"]`
- Started: `2026-11-07T14:37:21`
- Finished: `2026-11-07T14:42:03`
- Duration: `281.78s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.96s |  |
| generation #1 | PASS | 50.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 143.20s | best |
| profiling | PASS | 27.50s |  |
| artifacts | PASS | 43.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 282 --task-end 282
```
