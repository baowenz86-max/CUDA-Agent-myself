# CUDA Agent Workflow Report

- Task ID: `343`
- Status: **FAILED**
- Operators: `["nn.AdaptiveMaxPool1d"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:02:20`
- Duration: `48.06s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.24s |  |
| generation #1 | FAIL | 28.73s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 343 --task-end 343
```
