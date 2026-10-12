# CUDA Agent Workflow Report

- Task ID: `22`
- Status: **SUCCESS**
- Operators: `["torch.norm", "F.interpolate", "torch.is_complex"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:43:38`
- Duration: `442.62s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.77s |  |
| generation #1 | PASS | 51.22s | DeepSeek API |
| sources candidate #1 | FAIL | 0.04s | best |
| compile candidate #2 | FAIL | 70.81s | best |
| verification candidate #3 | PASS | 159.11s | best |
| profiling | PASS | 21.32s |  |
| artifacts | PASS | 97.12s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 22 --task-end 22
```
