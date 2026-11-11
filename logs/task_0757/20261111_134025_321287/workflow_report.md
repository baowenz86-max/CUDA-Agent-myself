# CUDA Agent Workflow Report

- Task ID: `757`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.ne"]`
- Started: `2026-11-11T13:40:25`
- Finished: `2026-11-11T13:43:34`
- Duration: `189.49s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 46.48s |  |
| generation #1 | PASS | 42.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 60.51s | best; initial_candidate |
| profiling | PASS | 38.56s |  |
| artifacts | PASS | 1.71s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 757 --task-end 757
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
