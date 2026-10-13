# CUDA Agent Workflow Report

- Task ID: `44`
- Status: **SUCCESS**
- Operators: `["F.leaky_relu"]`
- Started: `2026-10-13T15:21:18`
- Finished: `2026-10-13T15:24:32`
- Duration: `193.95s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.98s |  |
| generation #1 | PASS | 83.45s | DeepSeek API |
| compile candidate #1 | FAIL | 9.52s | best |
| verification candidate #2 | PASS | 34.93s | best |
| profiling | PASS | 16.39s |  |
| artifacts | PASS | 1.16s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 44 --task-end 44
```
