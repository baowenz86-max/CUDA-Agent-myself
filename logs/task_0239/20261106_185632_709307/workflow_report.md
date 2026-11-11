# CUDA Agent Workflow Report

- Task ID: `239`
- Status: **SUCCESS**
- Operators: `["nn.AvgPool3d", "nn.AvgPool3d"]`
- Started: `2026-11-06T18:56:32`
- Finished: `2026-11-06T19:02:39`
- Duration: `366.76s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.79s |  |
| generation #1 | PASS | 89.84s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 182.72s | best |
| profiling | PASS | 31.63s |  |
| artifacts | PASS | 50.70s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 239 --task-end 239
```
