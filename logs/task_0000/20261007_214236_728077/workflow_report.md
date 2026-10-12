# CUDA Agent Workflow Report

- Task ID: `0`
- Status: **FAILED**
- Operators: `["nn.BatchNorm3d", "torch.diag", "torch.max", "nn.Parameter"]`
- Started: `2026-10-07T21:42:36`
- Finished: `2026-10-07T21:46:15`
- Duration: `219.23s`
- Full log: `workflow.log`

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| generation #1 | FAIL | 106.99s | school API |
| generation #2 | FAIL | 106.00s | school API |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-id 0
```
