# CUDA Agent Workflow Report

- Task ID: `46`
- Status: **SUCCESS**
- Operators: `["F.interpolate", "torch.where", "nn.Upsample"]`
- Started: `2026-10-13T15:09:05`
- Finished: `2026-10-13T15:14:04`
- Duration: `298.59s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 48.34s |  |
| generation #1 | PASS | 63.81s | DeepSeek API |
| verification candidate #1 | PASS | 118.82s | best |
| profiling | PASS | 21.49s |  |
| artifacts | PASS | 46.05s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 46 --task-end 46
```
