# CUDA Agent Workflow Report

- Task ID: `146`
- Status: **FAILED**
- Operators: `["torch.ceil", "F.interpolate"]`
- Started: `2026-11-06T17:11:54`
- Finished: `2026-11-06T17:13:37`
- Duration: `103.48s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.70s |  |
| generation #1 | FAIL | 57.45s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 146 --task-end 146
```
