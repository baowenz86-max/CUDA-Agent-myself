# CUDA Agent Workflow Report

- Task ID: `193`
- Status: **SUCCESS**
- Operators: `["nn.GroupNorm", "torch.mean"]`
- Started: `2026-11-06T18:17:19`
- Finished: `2026-11-06T18:23:07`
- Duration: `348.00s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.68s |  |
| generation #1 | PASS | 39.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 192.36s | best |
| profiling | PASS | 54.60s |  |
| artifacts | PASS | 39.77s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 193 --task-end 193
```
