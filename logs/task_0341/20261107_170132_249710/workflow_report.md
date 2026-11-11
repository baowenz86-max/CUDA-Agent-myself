# CUDA Agent Workflow Report

- Task ID: `341`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.AvgPool1d"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:07:16`
- Duration: `344.39s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.08s |  |
| generation #1 | PASS | 96.96s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 201.29s | best |
| profiling | PASS | 28.25s |  |
| artifacts | PASS | 1.73s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 341 --task-end 341
```
