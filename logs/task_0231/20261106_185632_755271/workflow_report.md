# CUDA Agent Workflow Report

- Task ID: `231`
- Status: **SUCCESS**
- Operators: `["F.hardtanh", "torch.mean"]`
- Started: `2026-11-06T18:56:32`
- Finished: `2026-11-06T19:02:37`
- Duration: `364.30s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.31s |  |
| generation #1 | PASS | 41.65s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 95.33s | best |
| profiling | PASS | 32.00s |  |
| artifacts | PASS | 160.90s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 231 --task-end 231
```
