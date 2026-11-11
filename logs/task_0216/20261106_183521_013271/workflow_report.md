# CUDA Agent Workflow Report

- Task ID: `216`
- Status: **SUCCESS**
- Operators: `["F.gelu", "F.hardswish"]`
- Started: `2026-11-06T18:35:21`
- Finished: `2026-11-06T18:40:44`
- Duration: `323.78s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.65s |  |
| generation #1 | PASS | 34.53s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 117.50s | best |
| profiling | PASS | 70.31s |  |
| artifacts | PASS | 86.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 216 --task-end 216
```
