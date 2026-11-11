# CUDA Agent Workflow Report

- Task ID: `295`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "F.hardswish"]`
- Started: `2026-11-07T14:45:29`
- Finished: `2026-11-07T14:51:40`
- Duration: `371.09s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 41.73s |  |
| generation #1 | PASS | 46.99s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 246.25s | best |
| profiling | PASS | 32.52s |  |
| artifacts | PASS | 3.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 295 --task-end 295
```
