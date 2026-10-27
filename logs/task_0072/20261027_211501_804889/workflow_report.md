# CUDA Agent Workflow Report

- Task ID: `72`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.gelu"]`
- Started: `2026-10-27T21:15:01`
- Finished: `2026-10-27T21:22:12`
- Duration: `430.58s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 82.05s |  |
| generation #1 | PASS | 52.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 246.07s | best |
| profiling | PASS | 47.54s |  |
| artifacts | PASS | 1.79s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 72 --task-end 72
```
