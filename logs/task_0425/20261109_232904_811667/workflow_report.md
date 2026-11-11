# CUDA Agent Workflow Report

- Task ID: `425`
- Status: **SUCCESS**
- Operators: `["torch.max", "torch.triu"]`
- Started: `2026-11-09T23:29:04`
- Finished: `2026-11-09T23:32:22`
- Duration: `197.57s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.83s |  |
| generation #1 | PASS | 49.66s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 52.27s | best; initial_candidate |
| profiling | PASS | 22.26s |  |
| artifacts | PASS | 69.50s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 425 --task-end 425
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
