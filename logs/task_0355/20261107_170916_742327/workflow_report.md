# CUDA Agent Workflow Report

- Task ID: `355`
- Status: **SUCCESS**
- Operators: `["torch.native_batch_norm"]`
- Started: `2026-11-07T17:09:16`
- Finished: `2026-11-07T17:15:24`
- Duration: `367.68s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.99s |  |
| generation #1 | PASS | 41.85s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 137.18s | best |
| profiling | PASS | 48.79s |  |
| artifacts | PASS | 106.79s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 355 --task-end 355
```
