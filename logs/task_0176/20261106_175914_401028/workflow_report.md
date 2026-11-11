# CUDA Agent Workflow Report

- Task ID: `176`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "torch.sum"]`
- Started: `2026-11-06T17:59:14`
- Finished: `2026-11-06T18:04:44`
- Duration: `329.75s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.29s |  |
| generation #1 | PASS | 73.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 159.37s | best |
| profiling | PASS | 38.36s |  |
| artifacts | PASS | 23.14s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 176 --task-end 176
```
