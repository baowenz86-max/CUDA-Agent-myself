# CUDA Agent Workflow Report

- Task ID: `305`
- Status: **FAILED**
- Operators: `["nn.LPPool1d"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:36:05`
- Duration: `91.80s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 56.76s |  |
| generation #1 | FAIL | 34.93s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 305 --task-end 305
```
