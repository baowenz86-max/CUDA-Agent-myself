# CUDA Agent Workflow Report

- Task ID: `230`
- Status: **SUCCESS**
- Operators: `["torch.selu", "torch.einsum"]`
- Started: `2026-11-06T18:46:58`
- Finished: `2026-11-06T18:53:04`
- Duration: `366.54s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.80s |  |
| generation #1 | PASS | 49.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 205.67s | best |
| profiling | PASS | 56.26s |  |
| artifacts | PASS | 30.21s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 230 --task-end 230
```
