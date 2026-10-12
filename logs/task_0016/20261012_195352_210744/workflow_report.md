# CUDA Agent Workflow Report

- Task ID: `16`
- Status: **SUCCESS**
- Operators: `["torch.addmm", "torch.unsqueeze", "F.interpolate"]`
- Started: `2026-10-12T19:53:52`
- Finished: `2026-10-12T20:01:38`
- Duration: `466.16s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0006`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.70s |  |
| generation #1 | FAIL | 85.43s | DeepSeek API |
| generation #2 | PASS | 52.77s | DeepSeek API |
| verification candidate #2 | FAIL | 36.92s | best |
| verification candidate #3 | FAIL | 25.83s | rejected |
| verification candidate #4 | FAIL | 25.36s | rejected |
| generation #3 | PASS | 58.21s | DeepSeek API |
| verification candidate #5 | FAIL | 32.55s | rejected |
| verification candidate #6 | PASS | 23.05s | best |
| profiling | PASS | 10.78s |  |
| artifacts | PASS | 1.12s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 16 --task-end 16
```
