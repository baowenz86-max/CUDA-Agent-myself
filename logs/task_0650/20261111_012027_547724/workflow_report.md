# CUDA Agent Workflow Report

- Task ID: `650`
- Status: **SUCCESS**
- Operators: `["torch.isnan", "torch.gather"]`
- Started: `2026-11-11T01:20:27`
- Finished: `2026-11-11T01:23:43`
- Duration: `196.32s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.01s |  |
| generation #1 | PASS | 55.34s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 79.88s | best; initial_candidate |
| profiling | PASS | 28.46s |  |
| artifacts | PASS | 25.58s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 650 --task-end 650
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
