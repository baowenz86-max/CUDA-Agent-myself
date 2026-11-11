# CUDA Agent Workflow Report

- Task ID: `304`
- Status: **FAILED**
- Operators: `["torch.view_as_real", "nn.SyncBatchNorm"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:36:16`
- Duration: `102.90s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 53.93s |  |
| generation #1 | FAIL | 48.85s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 304 --task-end 304
```
