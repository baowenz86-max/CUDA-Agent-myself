# CUDA Agent Workflow Report

- Task ID: `196`
- Status: **SUCCESS**
- Operators: `["torch.tanh", "F.gelu"]`
- Started: `2026-11-06T18:17:19`
- Finished: `2026-11-06T18:23:56`
- Duration: `396.29s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 46.57s |  |
| generation #1 | PASS | 46.06s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 277.34s | best |
| profiling | PASS | 24.54s |  |
| artifacts | PASS | 1.70s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 196 --task-end 196
```
