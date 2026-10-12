# CUDA Agent Workflow Report

- Task ID: `12`
- Status: **SUCCESS**
- Operators: `["torch.sqrt", "nn.MultiheadAttention", "F.silu"]`
- Started: `2026-10-12T18:50:51`
- Finished: `2026-10-12T19:02:25`
- Duration: `694.42s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0005`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.17s |  |
| generation #1 | PASS | 104.31s | DeepSeek API |
| sources candidate #1 | FAIL | 0.06s | best |
| compile candidate #2 | FAIL | 26.89s | best |
| verification candidate #3 | FAIL | 221.39s | best |
| generation #2 | PASS | 93.41s | DeepSeek API |
| verification candidate #4 | FAIL | 27.31s | best |
| verification candidate #5 | PASS | 27.38s | best |
| profiling | PASS | 18.85s |  |
| artifacts | PASS | 1.39s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 12 --task-end 12
```
