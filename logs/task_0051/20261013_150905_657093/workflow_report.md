# CUDA Agent Workflow Report

- Task ID: `51`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.ReflectionPad2d", "torch.sub"]`
- Started: `2026-10-13T15:09:05`
- Finished: `2026-10-13T15:12:23`
- Duration: `197.67s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.84s |  |
| generation #1 | PASS | 70.85s | DeepSeek API |
| verification candidate #1 | PASS | 33.85s | best |
| profiling | PASS | 17.28s |  |
| artifacts | PASS | 72.79s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 51 --task-end 51
```
