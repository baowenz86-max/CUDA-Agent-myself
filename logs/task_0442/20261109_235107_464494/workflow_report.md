# CUDA Agent Workflow Report

- Task ID: `442`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.pad"]`
- Started: `2026-11-09T23:51:07`
- Finished: `2026-11-09T23:54:10`
- Duration: `183.04s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.70s |  |
| generation #1 | PASS | 49.06s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 90.35s | best; initial_candidate |
| profiling | PASS | 24.69s |  |
| artifacts | PASS | 10.19s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 442 --task-end 442
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
