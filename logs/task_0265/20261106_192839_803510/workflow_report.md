# CUDA Agent Workflow Report

- Task ID: `265`
- Status: **SUCCESS**
- Operators: `["torch.addcdiv"]`
- Started: `2026-11-06T19:28:39`
- Finished: `2026-11-06T19:33:32`
- Duration: `292.57s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.18s |  |
| generation #1 | PASS | 51.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 70.37s | best |
| profiling | PASS | 35.41s |  |
| artifacts | PASS | 127.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 265 --task-end 265
```
