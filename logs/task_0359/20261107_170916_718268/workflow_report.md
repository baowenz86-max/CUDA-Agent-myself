# CUDA Agent Workflow Report

- Task ID: `359`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.index_select"]`
- Started: `2026-11-07T17:09:16`
- Finished: `2026-11-07T17:15:04`
- Duration: `348.25s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.64s |  |
| generation #1 | PASS | 62.48s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 179.37s | best |
| profiling | PASS | 15.40s |  |
| artifacts | PASS | 67.26s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 359 --task-end 359
```
