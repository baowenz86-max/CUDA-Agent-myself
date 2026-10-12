# CUDA Agent Workflow Report

- Task ID: `11`
- Status: **SUCCESS**
- Operators: `["nn.Tanh", "nn.LeakyReLU", "nn.BatchNorm3d"]`
- Started: `2026-10-12T18:50:51`
- Finished: `2026-10-12T18:53:33`
- Duration: `161.91s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 15.57s |  |
| generation #1 | PASS | 32.67s | DeepSeek API |
| verification candidate #1 | PASS | 38.74s | best |
| profiling | PASS | 16.84s |  |
| artifacts | PASS | 58.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 11 --task-end 11
```
