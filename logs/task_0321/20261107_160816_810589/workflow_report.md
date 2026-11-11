# CUDA Agent Workflow Report

- Task ID: `321`
- Status: **SUCCESS**
- Operators: `["torch.sigmoid", "F.cosine_similarity"]`
- Started: `2026-11-07T16:08:16`
- Finished: `2026-11-07T16:13:02`
- Duration: `285.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.53s |  |
| generation #1 | PASS | 41.15s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 148.25s | best |
| profiling | PASS | 86.32s |  |
| artifacts | PASS | 1.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 321 --task-end 321
```
