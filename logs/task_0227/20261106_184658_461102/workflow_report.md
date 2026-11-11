# CUDA Agent Workflow Report

- Task ID: `227`
- Status: **SUCCESS**
- Operators: `["nn.Hardswish"]`
- Started: `2026-11-06T18:46:58`
- Finished: `2026-11-06T18:51:40`
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
| reference contract | PASS | 36.63s |  |
| generation #1 | PASS | 25.02s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 92.25s | best |
| profiling | PASS | 34.16s |  |
| artifacts | PASS | 93.66s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 227 --task-end 227
```
