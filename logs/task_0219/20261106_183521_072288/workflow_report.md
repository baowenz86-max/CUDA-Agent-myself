# CUDA Agent Workflow Report

- Task ID: `219`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.softplus"]`
- Started: `2026-11-06T18:35:21`
- Finished: `2026-11-06T18:42:53`
- Duration: `452.68s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 50.27s |  |
| generation #1 | PASS | 62.38s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 288.01s | best |
| profiling | PASS | 50.19s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 219 --task-end 219
```
