# CUDA Agent Workflow Report

- Task ID: `278`
- Status: **SUCCESS**
- Operators: `["F.elu", "nn.MaxPool2d"]`
- Started: `2026-11-07T13:54:42`
- Finished: `2026-11-07T14:01:19`
- Duration: `396.32s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 42.03s |  |
| generation #1 | PASS | 43.08s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 117.78s | best |
| profiling | PASS | 81.58s |  |
| artifacts | PASS | 111.55s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 278 --task-end 278
```
