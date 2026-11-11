# CUDA Agent Workflow Report

- Task ID: `156`
- Status: **SUCCESS**
- Operators: `["nn.Unfold", "nn.ConstantPad1d"]`
- Started: `2026-11-06T17:29:04`
- Finished: `2026-11-06T17:34:21`
- Duration: `317.08s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.07s |  |
| generation #1 | PASS | 82.61s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 137.20s | best |
| profiling | PASS | 16.72s |  |
| artifacts | PASS | 61.41s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 156 --task-end 156
```
