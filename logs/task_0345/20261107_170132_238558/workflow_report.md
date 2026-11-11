# CUDA Agent Workflow Report

- Task ID: `345`
- Status: **SUCCESS**
- Operators: `["F.unfold", "nn.BatchNorm1d"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:07:05`
- Duration: `332.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.65s |  |
| generation #1 | PASS | 88.83s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 160.29s | best |
| profiling | PASS | 46.91s |  |
| artifacts | PASS | 23.19s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 345 --task-end 345
```
