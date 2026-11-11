# CUDA Agent Workflow Report

- Task ID: `252`
- Status: **SUCCESS**
- Operators: `["torch.softmax", "F.elu"]`
- Started: `2026-11-06T19:14:13`
- Finished: `2026-11-06T19:18:59`
- Duration: `285.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.72s |  |
| generation #1 | PASS | 41.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 45.84s | best |
| profiling | PASS | 21.70s |  |
| artifacts | PASS | 170.08s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 252 --task-end 252
```
