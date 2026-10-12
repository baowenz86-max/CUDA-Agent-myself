# CUDA Agent Workflow Report

- Task ID: `0`
- Status: **FAILED**
- Operators: `["nn.BatchNorm3d", "torch.diag", "torch.max", "nn.Parameter"]`
- Started: `2026-09-28T21:35:57`
- Finished: `2026-09-28T21:41:50`
- Duration: `353.25s`
- Full log: `workflow.log`

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| generation #1 | FAIL | 120.12s | school API |
| generation #2 | FAIL | 120.02s | school API |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-id 0
```
