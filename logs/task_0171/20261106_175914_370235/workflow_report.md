# CUDA Agent Workflow Report

- Task ID: `171`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "F.elu"]`
- Started: `2026-11-06T17:59:14`
- Finished: `2026-11-06T18:04:11`
- Duration: `296.72s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.66s |  |
| generation #1 | PASS | 57.80s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 81.96s | best |
| profiling | PASS | 41.95s |  |
| artifacts | PASS | 89.27s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 171 --task-end 171
```
