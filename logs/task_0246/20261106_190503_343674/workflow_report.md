# CUDA Agent Workflow Report

- Task ID: `246`
- Status: **SUCCESS**
- Operators: `["nn.PixelShuffle"]`
- Started: `2026-11-06T19:05:03`
- Finished: `2026-11-06T19:09:49`
- Duration: `286.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 44.20s |  |
| generation #1 | PASS | 32.94s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 130.37s | best |
| profiling | PASS | 31.34s |  |
| artifacts | PASS | 47.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 246 --task-end 246
```
