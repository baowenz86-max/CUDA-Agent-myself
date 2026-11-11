# CUDA Agent Workflow Report

- Task ID: `323`
- Status: **SUCCESS**
- Operators: `["torch.relu"]`
- Started: `2026-11-07T16:08:16`
- Finished: `2026-11-07T16:12:20`
- Duration: `243.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.66s |  |
| generation #1 | PASS | 24.77s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 109.45s | best |
| profiling | PASS | 31.94s |  |
| artifacts | PASS | 62.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 323 --task-end 323
```
