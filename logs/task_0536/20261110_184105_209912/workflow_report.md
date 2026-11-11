# CUDA Agent Workflow Report

- Task ID: `536`
- Status: **SUCCESS**
- Operators: `["nn.CELU"]`
- Started: `2026-11-10T18:41:05`
- Finished: `2026-11-10T18:43:46`
- Duration: `161.50s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.89s |  |
| generation #1 | PASS | 69.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 48.44s | best; initial_candidate |
| profiling | PASS | 10.71s |  |
| artifacts | PASS | 1.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 536 --task-end 536
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
