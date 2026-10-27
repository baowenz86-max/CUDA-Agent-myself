# CUDA Agent Workflow Report

- Task ID: `129`
- Status: **SUCCESS**
- Operators: `["torch.softmax", "torch.cumprod"]`
- Started: `2026-10-27T20:20:36`
- Finished: `2026-10-27T20:25:11`
- Duration: `275.29s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 47.18s |  |
| generation #1 | PASS | 58.10s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 61.16s | best |
| profiling | PASS | 20.51s |  |
| artifacts | PASS | 88.28s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 129 --task-end 129
```
