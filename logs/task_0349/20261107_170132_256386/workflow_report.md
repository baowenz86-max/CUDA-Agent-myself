# CUDA Agent Workflow Report

- Task ID: `349`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.Hardswish"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:07:03`
- Duration: `331.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.65s |  |
| generation #1 | PASS | 58.53s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 167.15s | best |
| profiling | PASS | 36.72s |  |
| artifacts | PASS | 41.47s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 349 --task-end 349
```
