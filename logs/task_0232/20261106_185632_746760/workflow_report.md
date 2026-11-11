# CUDA Agent Workflow Report

- Task ID: `232`
- Status: **SUCCESS**
- Operators: `["nn.InstanceNorm2d", "mul"]`
- Started: `2026-11-06T18:56:32`
- Finished: `2026-11-06T19:02:41`
- Duration: `368.29s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.99s |  |
| generation #1 | PASS | 86.14s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 222.68s | best |
| profiling | PASS | 25.32s |  |
| artifacts | PASS | 11.09s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 232 --task-end 232
```
