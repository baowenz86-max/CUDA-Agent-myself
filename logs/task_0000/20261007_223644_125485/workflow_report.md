# CUDA Agent Workflow Report

- Task ID: `0`
- Status: **FAILED**
- Operators: `["nn.BatchNorm3d", "torch.diag", "torch.max", "nn.Parameter"]`
- Started: `2026-10-07T22:36:44`
- Finished: `2026-10-07T22:42:53`
- Duration: `369.72s`
- Full log: `workflow.log`

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| generation #1 | PASS | 24.27s | school API |
| compile | PASS | 88.04s |  |
| verification | FAIL | 55.73s |  |
| generation #2 | PASS | 17.27s | school API |
| compile | PASS | 20.61s |  |
| verification | FAIL | 20.44s |  |
| generation #3 | PASS | 18.12s | school API |
| compile | PASS | 20.60s |  |
| verification | FAIL | 39.87s |  |
| generation #4 | PASS | 18.02s | school API |
| compile | PASS | 20.53s |  |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-id 0
```
