# CUDA Agent Workflow Report

- Task ID: `587`
- Status: **SUCCESS**
- Operators: `["torch.isinf", "nn.GroupNorm"]`
- Started: `2026-11-10T20:36:03`
- Finished: `2026-11-10T20:40:06`
- Duration: `242.56s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.35s |  |
| generation #1 | PASS | 75.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 124.29s | best; initial_candidate |
| profiling | PASS | 19.96s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 587 --task-end 587
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
