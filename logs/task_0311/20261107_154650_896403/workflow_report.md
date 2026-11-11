# CUDA Agent Workflow Report

- Task ID: `311`
- Status: **SUCCESS**
- Operators: `["F.leaky_relu", "F.leaky_relu"]`
- Started: `2026-11-07T15:46:50`
- Finished: `2026-11-07T16:00:22`
- Duration: `811.12s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.11s |  |
| generation #1 | PASS | 586.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 47.17s | best |
| profiling | PASS | 22.73s |  |
| artifacts | PASS | 150.84s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 311 --task-end 311
```
