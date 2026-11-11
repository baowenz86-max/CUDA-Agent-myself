# CUDA Agent Workflow Report

- Task ID: `306`
- Status: **FAILED**
- Operators: `["F.softplus", "nn.Conv3d"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:36:02`
- Duration: `88.70s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 38.60s |  |
| generation #1 | FAIL | 49.97s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 306 --task-end 306
```
