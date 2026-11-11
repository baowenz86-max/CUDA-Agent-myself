# CUDA Agent Workflow Report

- Task ID: `388`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.BCEWithLogitsLoss"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:42:30`
- Duration: `429.66s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 50.67s |  |
| generation #1 | PASS | 137.60s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 188.04s | best; initial_candidate |
| profiling | PASS | 31.76s |  |
| artifacts | PASS | 21.22s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 388 --task-end 388
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
