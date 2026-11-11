# CUDA Agent Workflow Report

- Task ID: `336`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.squeeze"]`
- Started: `2026-11-07T16:17:25`
- Finished: `2026-11-07T16:21:04`
- Duration: `219.38s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.42s |  |
| generation #1 | PASS | 31.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 49.78s | best |
| profiling | PASS | 14.65s |  |
| artifacts | PASS | 101.80s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 336 --task-end 336
```
