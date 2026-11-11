# CUDA Agent Workflow Report

- Task ID: `298`
- Status: **SUCCESS**
- Operators: `["nn.LayerNorm", "F.gelu"]`
- Started: `2026-11-07T14:45:29`
- Finished: `2026-11-07T14:51:39`
- Duration: `369.92s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.27s |  |
| generation #1 | PASS | 24.57s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 204.06s | best |
| profiling | PASS | 54.34s |  |
| artifacts | PASS | 58.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 298 --task-end 298
```
