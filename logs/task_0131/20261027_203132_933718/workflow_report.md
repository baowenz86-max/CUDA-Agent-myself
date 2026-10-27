# CUDA Agent Workflow Report

- Task ID: `131`
- Status: **SUCCESS**
- Operators: `["torch.relu", "nn.GELU"]`
- Started: `2026-10-27T20:31:32`
- Finished: `2026-10-27T20:36:35`
- Duration: `303.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.78s |  |
| generation #1 | PASS | 34.60s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 122.34s | best |
| profiling | PASS | 51.78s |  |
| artifacts | PASS | 66.48s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 131 --task-end 131
```
