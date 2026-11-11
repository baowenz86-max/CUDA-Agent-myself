# CUDA Agent Workflow Report

- Task ID: `301`
- Status: **FAILED**
- Operators: `["torch.matmul"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:36:04`
- Duration: `90.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.70s |  |
| generation #1 | FAIL | 82.61s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 301 --task-end 301
```
