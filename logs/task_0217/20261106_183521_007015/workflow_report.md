# CUDA Agent Workflow Report

- Task ID: `217`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "mul"]`
- Started: `2026-11-06T18:35:21`
- Finished: `2026-11-06T18:41:44`
- Duration: `383.64s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.84s |  |
| generation #1 | PASS | 61.40s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 176.85s | best |
| profiling | PASS | 83.53s |  |
| artifacts | PASS | 51.92s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 217 --task-end 217
```
