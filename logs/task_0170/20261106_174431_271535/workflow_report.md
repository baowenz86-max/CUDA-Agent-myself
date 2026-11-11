# CUDA Agent Workflow Report

- Task ID: `170`
- Status: **SUCCESS**
- Operators: `["nn.AvgPool2d", "nn.Hardtanh"]`
- Started: `2026-11-06T17:44:31`
- Finished: `2026-11-06T17:50:08`
- Duration: `337.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 39.65s |  |
| generation #1 | PASS | 67.81s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 161.40s | best |
| profiling | PASS | 35.49s |  |
| artifacts | PASS | 33.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 170 --task-end 170
```
