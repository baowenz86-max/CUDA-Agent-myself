# CUDA Agent Workflow Report

- Task ID: `257`
- Status: **SUCCESS**
- Operators: `["torch.affine_grid_generator"]`
- Started: `2026-11-06T19:14:14`
- Finished: `2026-11-06T19:19:17`
- Duration: `303.10s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.53s |  |
| generation #1 | PASS | 43.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 60.91s | best |
| profiling | PASS | 20.93s |  |
| artifacts | PASS | 159.30s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 257 --task-end 257
```
