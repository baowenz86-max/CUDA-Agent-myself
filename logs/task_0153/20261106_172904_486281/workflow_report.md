# CUDA Agent Workflow Report

- Task ID: `153`
- Status: **SUCCESS**
- Operators: `["nn.InstanceNorm2d", "add"]`
- Started: `2026-11-06T17:29:04`
- Finished: `2026-11-06T17:34:51`
- Duration: `346.93s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.32s |  |
| generation #1 | PASS | 64.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 39.70s | best |
| profiling | PASS | 21.22s |  |
| artifacts | PASS | 216.70s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 153 --task-end 153
```
