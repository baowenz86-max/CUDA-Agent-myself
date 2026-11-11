# CUDA Agent Workflow Report

- Task ID: `730`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.sign"]`
- Started: `2026-11-11T12:55:49`
- Finished: `2026-11-11T13:00:07`
- Duration: `258.49s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 44.92s |  |
| generation #1 | PASS | 25.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 150.66s | best; initial_candidate |
| profiling | PASS | 35.62s |  |
| artifacts | PASS | 1.98s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 730 --task-end 730
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
