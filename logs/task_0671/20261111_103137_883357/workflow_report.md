# CUDA Agent Workflow Report

- Task ID: `671`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.floor"]`
- Started: `2026-11-11T10:31:37`
- Finished: `2026-11-11T10:33:49`
- Duration: `131.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.92s |  |
| generation #1 | PASS | 38.43s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 53.22s | best; initial_candidate |
| profiling | PASS | 30.21s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 671 --task-end 671
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
