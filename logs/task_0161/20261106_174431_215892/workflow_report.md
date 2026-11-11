# CUDA Agent Workflow Report

- Task ID: `161`
- Status: **SUCCESS**
- Operators: `["nn.AvgPool3d", "torch.sum"]`
- Started: `2026-11-06T17:44:31`
- Finished: `2026-11-06T17:49:30`
- Duration: `299.08s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.55s |  |
| generation #1 | PASS | 65.55s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 122.90s | best |
| profiling | PASS | 29.17s |  |
| artifacts | PASS | 76.86s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 161 --task-end 161
```
