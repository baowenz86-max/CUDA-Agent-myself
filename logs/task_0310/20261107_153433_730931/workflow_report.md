# CUDA Agent Workflow Report

- Task ID: `310`
- Status: **FAILED**
- Operators: `["nn.InstanceNorm2d", "torch.selu"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:36:19`
- Duration: `106.07s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.61s |  |
| generation #1 | FAIL | 82.34s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 310 --task-end 310
```
