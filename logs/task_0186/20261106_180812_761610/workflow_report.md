# CUDA Agent Workflow Report

- Task ID: `186`
- Status: **SUCCESS**
- Operators: `["torch.mean", "mul"]`
- Started: `2026-11-06T18:08:12`
- Finished: `2026-11-06T18:14:11`
- Duration: `359.23s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.55s |  |
| generation #1 | PASS | 58.88s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 115.13s | best |
| profiling | PASS | 24.63s |  |
| artifacts | PASS | 138.94s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 186 --task-end 186
```
