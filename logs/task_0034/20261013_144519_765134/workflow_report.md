# CUDA Agent Workflow Report

- Task ID: `34`
- Status: **SUCCESS**
- Operators: `["F.one_hot", "nn.ConvTranspose2d", "nn.Upsample"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T14:52:57`
- Duration: `457.82s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.71s |  |
| generation #1 | PASS | 209.79s | DeepSeek API |
| compile candidate #1 | FAIL | 44.64s | best |
| interface candidate #2 | FAIL | 54.83s | best |
| verification candidate #3 | PASS | 35.97s | best |
| profiling | PASS | 11.88s |  |
| artifacts | PASS | 1.11s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 34 --task-end 34
```
