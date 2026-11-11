# CUDA Agent Workflow Report

- Task ID: `309`
- Status: **FAILED**
- Operators: `["nn.GroupNorm", "torch.softmax"]`
- Started: `2026-11-07T15:34:33`
- Finished: `2026-11-07T15:35:49`
- Duration: `75.27s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 59.87s |  |
| generation #1 | FAIL | 15.29s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 309 --task-end 309
```
