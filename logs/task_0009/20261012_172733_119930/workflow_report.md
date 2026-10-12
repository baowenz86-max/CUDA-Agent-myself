# CUDA Agent Workflow Report

- Task ID: `9`
- Status: **SUCCESS**
- Operators: `["torch.linspace", "nn.ReflectionPad2d", "torch.histc"]`
- Started: `2026-10-12T17:27:33`
- Finished: `2026-10-12T17:45:39`
- Duration: `1086.30s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0006`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.54s |  |
| generation #1 | FAIL | 117.25s | DeepSeek API |
| generation #2 | PASS | 50.03s | DeepSeek API |
| sources candidate #2 | FAIL | 0.05s | rejected |
| compile candidate #3 | FAIL | 59.57s | best |
| verification candidate #4 | FAIL | 398.68s | best |
| generation #3 | PASS | 71.13s | DeepSeek API |
| compile candidate #5 | FAIL | 157.22s | rejected |
| verification candidate #6 | PASS | 131.14s | best |
| profiling | PASS | 13.94s |  |
| artifacts | PASS | 7.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 9 --task-end 9
```
