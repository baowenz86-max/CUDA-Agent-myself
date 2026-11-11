# CUDA Agent Workflow Report

- Task ID: `609`
- Status: **SUCCESS**
- Operators: `["torch.isnan", "torch.masked_fill"]`
- Started: `2026-11-11T00:00:34`
- Finished: `2026-11-11T00:03:23`
- Duration: `168.85s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.50s |  |
| generation #1 | PASS | 56.88s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 51.74s | best; initial_candidate |
| profiling | PASS | 31.11s |  |
| artifacts | PASS | 1.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 609 --task-end 609
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
