# CUDA Agent Workflow Report

- Task ID: `37`
- Status: **SUCCESS**
- Operators: `["F.one_hot", "F.pad", "F.softmax"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T14:52:38`
- Duration: `438.37s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.80s |  |
| generation #1 | PASS | 71.67s | DeepSeek API |
| verification candidate #1 | FAIL | 70.18s | best |
| verification candidate #2 | FAIL | 46.72s | best |
| verification candidate #3 | PASS | 44.63s | best |
| profiling | PASS | 17.58s |  |
| artifacts | PASS | 68.58s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 37 --task-end 37
```
