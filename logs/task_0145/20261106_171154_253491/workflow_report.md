# CUDA Agent Workflow Report

- Task ID: `145`
- Status: **SUCCESS**
- Operators: `["torch.tril", "nn.InstanceNorm2d"]`
- Started: `2026-11-06T17:11:54`
- Finished: `2026-11-06T17:18:24`
- Duration: `389.96s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 70.75s |  |
| generation #1 | PASS | 187.23s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 103.67s | best |
| profiling | PASS | 13.60s |  |
| artifacts | PASS | 14.34s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 145 --task-end 145
```
