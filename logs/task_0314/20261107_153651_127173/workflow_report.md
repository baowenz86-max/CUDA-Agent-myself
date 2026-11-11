# CUDA Agent Workflow Report

- Task ID: `314`
- Status: **FAILED**
- Operators: `["F.leaky_relu", "nn.AvgPool1d"]`
- Started: `2026-11-07T15:36:51`
- Finished: `2026-11-07T15:37:18`
- Duration: `27.30s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.69s |  |
| generation #1 | FAIL | 15.27s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 314 --task-end 314
```
