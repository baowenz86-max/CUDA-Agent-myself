# CUDA Agent Workflow Report

- Task ID: `26`
- Status: **SUCCESS**
- Operators: `["F.one_hot", "torch.maximum", "torch.bucketize"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:44:23`
- Duration: `487.10s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 38.26s |  |
| generation #1 | PASS | 40.68s | DeepSeek API |
| compile candidate #1 | FAIL | 95.34s | best |
| verification candidate #2 | FAIL | 153.97s | best |
| verification candidate #3 | PASS | 67.83s | best |
| profiling | PASS | 14.61s |  |
| artifacts | PASS | 22.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 26 --task-end 26
```
