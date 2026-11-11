# CUDA Agent Workflow Report

- Task ID: `200`
- Status: **SUCCESS**
- Operators: `["F.hardsigmoid", "F.elu"]`
- Started: `2026-11-06T18:17:19`
- Finished: `2026-11-06T18:23:09`
- Duration: `349.60s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 26.08s |  |
| generation #1 | PASS | 51.49s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 197.07s | best |
| profiling | PASS | 67.35s |  |
| artifacts | PASS | 7.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 200 --task-end 200
```
