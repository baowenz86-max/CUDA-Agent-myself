# CUDA Agent Workflow Report

- Task ID: `31`
- Status: **SUCCESS**
- Operators: `["torch.histc", "F.group_norm", "nn.BatchNorm2d"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T14:49:52`
- Duration: `272.91s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.70s |  |
| generation #1 | PASS | 82.59s | DeepSeek API |
| compile candidate #1 | FAIL | 31.95s | best |
| verification candidate #2 | PASS | 69.57s | best |
| profiling | PASS | 16.66s |  |
| artifacts | PASS | 37.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 31 --task-end 31
```
