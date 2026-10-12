# CUDA Agent Workflow Report

- Task ID: `7`
- Status: **SUCCESS**
- Operators: `["F.scaled_dot_product_attention", "nn.BCEWithLogitsLoss", "torch.baddbmm"]`
- Started: `2026-10-12T17:27:33`
- Finished: `2026-10-12T17:51:58`
- Duration: `1465.66s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0006`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.76s |  |
| generation #1 | PASS | 188.35s | DeepSeek API |
| compile candidate #1 | FAIL | 96.77s | best |
| interface candidate #2 | FAIL | 356.63s | best |
| verification candidate #3 | FAIL | 245.51s | rejected |
| generation #2 | PASS | 203.15s | DeepSeek API |
| compile candidate #4 | FAIL | 72.16s | rejected |
| verification candidate #5 | FAIL | 73.79s | rejected |
| verification candidate #6 | PASS | 139.13s | best |
| profiling | PASS | 18.42s |  |
| artifacts | PASS | 8.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 7 --task-end 7
```
