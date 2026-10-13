# CUDA Agent Workflow Report

- Task ID: `49`
- Status: **FAILED**
- Operators: `["F.conv_transpose1d", "nn.ParameterList", "torch.round"]`
- Started: `2026-10-13T15:09:05`
- Finished: `2026-10-13T15:14:28`
- Duration: `322.78s`
- Full log: `workflow.log`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.58s |  |
| generation #1 | FAIL | 313.56s | DeepSeek API |
| generation #2 | FAIL | 0.25s | DeepSeek API |
| generation #3 | FAIL | 0.23s | DeepSeek API |

## Failure Summary

Best candidate 1, stage sources: Generation failed; see workflow.log and api_calls.jsonl

## Reproduction

```bash
python main.py --task-start 49 --task-end 49
```
