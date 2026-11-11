# CUDA Agent Workflow Report

- Task ID: `326`
- Status: **SUCCESS**
- Operators: `["F.hardsigmoid", "torch.mean"]`
- Started: `2026-11-07T16:08:16`
- Finished: `2026-11-07T16:14:38`
- Duration: `381.20s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.81s |  |
| generation #1 | PASS | 63.49s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 185.54s | best |
| profiling | PASS | 47.31s |  |
| artifacts | PASS | 63.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 326 --task-end 326
```
