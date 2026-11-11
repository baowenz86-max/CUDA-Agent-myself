# CUDA Agent Workflow Report

- Task ID: `218`
- Status: **SUCCESS**
- Operators: `["F.leaky_relu", "nn.InstanceNorm2d"]`
- Started: `2026-11-06T18:35:21`
- Finished: `2026-11-06T18:41:45`
- Duration: `384.58s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 39.29s |  |
| generation #1 | PASS | 66.13s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 207.33s | best |
| profiling | PASS | 41.71s |  |
| artifacts | PASS | 30.00s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 218 --task-end 218
```
