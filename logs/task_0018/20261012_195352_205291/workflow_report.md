# CUDA Agent Workflow Report

- Task ID: `18`
- Status: **SUCCESS**
- Operators: `["nn.ReflectionPad2d", "nn.ReLU", "torch.index_select"]`
- Started: `2026-10-12T19:53:52`
- Finished: `2026-10-12T19:56:28`
- Duration: `156.20s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.07s |  |
| generation #1 | PASS | 42.32s | DeepSeek API |
| verification candidate #1 | PASS | 55.96s | best |
| profiling | PASS | 34.62s |  |
| artifacts | PASS | 2.05s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 18 --task-end 18
```
