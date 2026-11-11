# CUDA Agent Workflow Report

- Task ID: `372`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.hardswish"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:04:15`
- Duration: `531.34s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.58s |  |
| generation #1 | PASS | 91.10s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 380.41s | best; initial_candidate |
| profiling | PASS | 35.47s |  |
| artifacts | PASS | 1.73s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 372 --task-end 372
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
