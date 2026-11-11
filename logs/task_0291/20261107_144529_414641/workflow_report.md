# CUDA Agent Workflow Report

- Task ID: `291`
- Status: **SUCCESS**
- Operators: `["torch.triu", "nn.functional"]`
- Started: `2026-11-07T14:45:29`
- Finished: `2026-11-07T14:51:31`
- Duration: `361.72s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.19s |  |
| generation #1 | PASS | 29.05s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 158.50s | best |
| profiling | PASS | 75.22s |  |
| artifacts | PASS | 76.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 291 --task-end 291
```
