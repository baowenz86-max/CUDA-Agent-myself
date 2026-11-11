# CUDA Agent Workflow Report

- Task ID: `780`
- Status: **SUCCESS**
- Operators: `["torch.isin", "nn.ParameterList"]`
- Started: `2026-11-11T14:09:40`
- Finished: `2026-11-11T14:12:23`
- Duration: `162.91s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.30s |  |
| generation #1 | PASS | 45.90s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 35.45s | best; initial_candidate |
| profiling | PASS | 18.27s |  |
| artifacts | PASS | 52.94s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 780 --task-end 780
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
