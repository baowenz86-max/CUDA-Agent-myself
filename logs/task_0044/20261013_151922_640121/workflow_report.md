# CUDA Agent Workflow Report

- Task ID: `44`
- Status: **FAILED**
- Operators: `["F.leaky_relu"]`
- Started: `2026-10-13T15:19:22`
- Finished: `2026-10-13T15:19:37`
- Duration: `14.63s`
- Full log: `workflow.log`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.87s |  |
| generation #1 | FAIL | 0.60s | DeepSeek API |
| generation #2 | FAIL | 0.51s | DeepSeek API |
| generation #3 | FAIL | 0.58s | DeepSeek API |

## Failure Summary

Best candidate 1, stage sources: Generation failed; see workflow.log and api_calls.jsonl

## Reproduction

```bash
python main.py --task-start 44 --task-end 44
```
