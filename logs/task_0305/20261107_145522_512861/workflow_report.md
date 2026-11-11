# CUDA Agent Workflow Report

- Task ID: `305`
- Status: **SUCCESS**
- Operators: `["nn.LPPool1d"]`
- Started: `2026-11-07T14:55:22`
- Finished: `2026-11-07T14:59:08`
- Duration: `226.10s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.07s |  |
| generation #1 | PASS | 38.99s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 34.20s | best |
| profiling | PASS | 16.41s |  |
| artifacts | PASS | 124.37s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 305 --task-end 305
```
