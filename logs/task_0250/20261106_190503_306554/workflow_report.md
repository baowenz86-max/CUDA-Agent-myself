# CUDA Agent Workflow Report

- Task ID: `250`
- Status: **SUCCESS**
- Operators: `["F.hardsigmoid", "nn.functional"]`
- Started: `2026-11-06T19:05:03`
- Finished: `2026-11-06T19:10:14`
- Duration: `310.81s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.96s |  |
| generation #1 | PASS | 50.01s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 204.25s | best |
| profiling | PASS | 25.93s |  |
| artifacts | PASS | 1.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 250 --task-end 250
```
