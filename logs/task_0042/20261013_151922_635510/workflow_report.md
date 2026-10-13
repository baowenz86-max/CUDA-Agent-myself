# CUDA Agent Workflow Report

- Task ID: `42`
- Status: **FAILED**
- Operators: `["nn.MultiheadAttention", "torch.cat", "F.smooth_l1_loss"]`
- Started: `2026-10-13T15:19:22`
- Finished: `2026-10-13T15:19:34`
- Duration: `11.78s`
- Full log: `workflow.log`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.78s |  |
| generation #1 | FAIL | 0.63s | DeepSeek API |
| generation #2 | FAIL | 0.70s | DeepSeek API |
| generation #3 | FAIL | 0.59s | DeepSeek API |

## Failure Summary

Best candidate 1, stage sources: Generation failed; see workflow.log and api_calls.jsonl

## Reproduction

```bash
python main.py --task-start 42 --task-end 42
```
