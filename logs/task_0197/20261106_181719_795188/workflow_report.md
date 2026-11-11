# CUDA Agent Workflow Report

- Task ID: `197`
- Status: **SUCCESS**
- Operators: `["F.elu", "nn.AvgPool2d"]`
- Started: `2026-11-06T18:17:19`
- Finished: `2026-11-06T18:21:42`
- Duration: `262.84s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.44s |  |
| generation #1 | PASS | 50.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 42.46s | best |
| profiling | PASS | 21.72s |  |
| artifacts | PASS | 141.03s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 197 --task-end 197
```
