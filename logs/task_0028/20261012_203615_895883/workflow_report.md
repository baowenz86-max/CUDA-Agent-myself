# CUDA Agent Workflow Report

- Task ID: `28`
- Status: **SUCCESS**
- Operators: `["nn.Unfold", "torch.sign", "torch.cos"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:42:50`
- Duration: `394.18s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.76s |  |
| generation #1 | PASS | 42.53s | DeepSeek API |
| verification candidate #1 | PASS | 94.88s | best |
| profiling | PASS | 19.15s |  |
| artifacts | PASS | 204.78s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 28 --task-end 28
```
