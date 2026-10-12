# CUDA Agent Workflow Report

- Task ID: `23`
- Status: **SUCCESS**
- Operators: `["torch.median", "torch.meshgrid", "nn.ReflectionPad2d"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:42:55`
- Duration: `399.26s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.22s |  |
| generation #1 | PASS | 43.79s | DeepSeek API |
| sources candidate #1 | FAIL | 0.04s | best |
| verification candidate #2 | PASS | 156.17s | best |
| profiling | PASS | 14.74s |  |
| artifacts | PASS | 146.27s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 23 --task-end 23
```
