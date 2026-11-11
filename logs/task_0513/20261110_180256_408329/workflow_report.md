# CUDA Agent Workflow Report

- Task ID: `513`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool1d"]`
- Started: `2026-11-10T18:02:56`
- Finished: `2026-11-10T18:04:40`
- Duration: `104.24s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.86s |  |
| generation #1 | PASS | 43.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 37.15s | best; initial_candidate |
| profiling | PASS | 12.03s |  |
| artifacts | PASS | 1.56s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 513 --task-end 513
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
