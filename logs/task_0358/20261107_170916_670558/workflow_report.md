# CUDA Agent Workflow Report

- Task ID: `358`
- Status: **SUCCESS**
- Operators: `["F.adaptive_max_pool2d"]`
- Started: `2026-11-07T17:09:16`
- Finished: `2026-11-07T17:15:23`
- Duration: `366.70s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.46s |  |
| generation #1 | PASS | 69.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 124.96s | best |
| profiling | PASS | 36.94s |  |
| artifacts | PASS | 130.21s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 358 --task-end 358
```
