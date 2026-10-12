# CUDA Agent Workflow Report

- Task ID: `25`
- Status: **SUCCESS**
- Operators: `["torch.erf", "torch.sin", "nn.ReflectionPad2d"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:42:48`
- Duration: `393.00s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.60s |  |
| generation #1 | PASS | 37.48s | DeepSeek API |
| verification candidate #1 | PASS | 45.69s | best |
| profiling | PASS | 44.20s |  |
| artifacts | PASS | 229.95s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 25 --task-end 25
```
