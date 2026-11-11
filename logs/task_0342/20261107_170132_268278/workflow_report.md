# CUDA Agent Workflow Report

- Task ID: `342`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool1d"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:06:33`
- Duration: `301.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.32s |  |
| generation #1 | PASS | 41.49s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 90.30s | best |
| profiling | PASS | 21.06s |  |
| artifacts | PASS | 113.87s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 342 --task-end 342
```
