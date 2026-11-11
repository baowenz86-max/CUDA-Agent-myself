# CUDA Agent Workflow Report

- Task ID: `423`
- Status: **SUCCESS**
- Operators: `["torch.clamp", "torch.triu"]`
- Started: `2026-11-09T23:29:04`
- Finished: `2026-11-09T23:32:44`
- Duration: `219.93s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.90s |  |
| generation #1 | PASS | 43.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 109.04s | best; initial_candidate |
| profiling | PASS | 43.62s |  |
| artifacts | PASS | 1.78s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 423 --task-end 423
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
