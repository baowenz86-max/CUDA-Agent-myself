# CUDA Agent Workflow Report

- Task ID: `0`
- Status: **FAILED**
- Operators: `["nn.BatchNorm3d", "torch.diag", "torch.max", "nn.Parameter"]`
- Started: `2026-10-08T12:15:47`
- Finished: `2026-10-08T12:21:23`
- Duration: `336.51s`
- Full log: `workflow.log`

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| generation #1 | PASS | 18.73s | school API |
| compile | PASS | 49.65s |  |
| verification | FAIL | 30.74s |  |
| generation #2 | PASS | 13.91s | school API |
| compile | PASS | 20.14s |  |
| verification | FAIL | 24.19s |  |
| generation #3 | PASS | 13.92s | school API |
| compile | PASS | 20.88s |  |
| verification | FAIL | 24.11s |  |
| generation #4 | PASS | 13.71s | school API |
| compile | PASS | 20.07s |  |
| verification | FAIL | 23.83s |  |
| generation #5 | PASS | 13.76s | school API |
| compile | PASS | 20.15s |  |
| verification | FAIL | 28.64s |  |

## Failure Summary

All generation attempts were exhausted. See workflow.log.

## Reproduction

```bash
python main.py --task-id 0
```
