# CUDA Agent Workflow Report

- Task ID: `157`
- Status: **SUCCESS**
- Operators: `["torch.relu", "torch.selu"]`
- Started: `2026-11-06T17:29:04`
- Finished: `2026-11-06T17:35:17`
- Duration: `372.44s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.03s |  |
| generation #1 | PASS | 92.85s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 166.07s | best |
| profiling | PASS | 25.29s |  |
| artifacts | PASS | 60.12s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 157 --task-end 157
```
