# CUDA Agent Workflow Report

- Task ID: `303`
- Status: **FAILED**
- Operators: `["F.pixel_shuffle"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:35:53`
- Duration: `79.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 64.06s |  |
| generation #1 | FAIL | 15.43s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 303 --task-end 303
```
