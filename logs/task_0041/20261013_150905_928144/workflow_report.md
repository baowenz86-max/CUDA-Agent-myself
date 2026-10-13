# CUDA Agent Workflow Report

- Task ID: `41`
- Status: **SUCCESS**
- Operators: `["torch.neg", "nn.AdaptiveAvgPool1d", "torch.addcmul"]`
- Started: `2026-10-13T15:09:05`
- Finished: `2026-10-13T15:15:25`
- Duration: `379.34s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.38s |  |
| generation #1 | PASS | 75.62s | DeepSeek API |
| sources candidate #1 | FAIL | 0.03s | best |
| verification candidate #2 | FAIL | 127.30s | best |
| verification candidate #3 | PASS | 60.98s | best |
| profiling | PASS | 17.62s |  |
| artifacts | PASS | 28.48s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 41 --task-end 41
```
