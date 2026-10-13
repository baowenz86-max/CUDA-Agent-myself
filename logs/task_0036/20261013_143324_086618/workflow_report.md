# CUDA Agent Workflow Report

- Task ID: `36`
- Status: **SUCCESS**
- Operators: `["torch.add", "torch.argmin", "torch.stft"]`
- Started: `2026-10-13T14:33:24`
- Finished: `2026-10-13T14:40:49`
- Duration: `445.01s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 49.89s |  |
| generation #1 | PASS | 112.36s | DeepSeek API |
| verification candidate #1 | PASS | 123.24s | best |
| profiling | PASS | 48.71s |  |
| artifacts | PASS | 110.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 36 --task-end 36
```
