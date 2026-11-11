# CUDA Agent Workflow Report

- Task ID: `189`
- Status: **SUCCESS**
- Operators: `["torch.matmul", "nn.AdaptiveAvgPool3d"]`
- Started: `2026-11-06T18:08:12`
- Finished: `2026-11-06T18:13:23`
- Duration: `310.29s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.38s |  |
| generation #1 | PASS | 50.42s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 34.16s | best |
| profiling | PASS | 15.59s |  |
| artifacts | PASS | 200.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 189 --task-end 189
```
