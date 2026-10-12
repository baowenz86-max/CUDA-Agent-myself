# CUDA Agent Workflow Report

- Task ID: `30`
- Status: **SUCCESS**
- Operators: `["torch.polar", "F.one_hot", "F.scaled_dot_product_attention"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:44:21`
- Duration: `485.49s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 30.24s |  |
| generation #1 | PASS | 50.46s | DeepSeek API |
| verification candidate #1 | FAIL | 142.61s | best |
| verification candidate #2 | PASS | 123.97s | best |
| profiling | PASS | 31.91s |  |
| artifacts | PASS | 77.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 30 --task-end 30
```
