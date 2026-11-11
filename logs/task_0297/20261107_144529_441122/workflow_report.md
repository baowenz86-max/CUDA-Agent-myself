# CUDA Agent Workflow Report

- Task ID: `297`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "torch.triu"]`
- Started: `2026-11-07T14:45:29`
- Finished: `2026-11-07T14:52:29`
- Duration: `420.06s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 38.40s |  |
| generation #1 | PASS | 53.61s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 292.33s | best |
| profiling | PASS | 34.00s |  |
| artifacts | PASS | 1.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 297 --task-end 297
```
