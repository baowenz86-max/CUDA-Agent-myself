# CUDA Agent Workflow Report

- Task ID: `558`
- Status: **SUCCESS**
- Operators: `["F.hardsigmoid"]`
- Started: `2026-11-10T19:40:32`
- Finished: `2026-11-10T19:43:43`
- Duration: `190.91s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.09s |  |
| generation #1 | PASS | 30.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 101.89s | best; initial_candidate |
| profiling | PASS | 21.82s |  |
| artifacts | PASS | 1.52s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 558 --task-end 558
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
