# CUDA Agent Workflow Report

- Task ID: `626`
- Status: **SUCCESS**
- Operators: `["torch.sum", "F.softplus"]`
- Started: `2026-11-11T00:24:17`
- Finished: `2026-11-11T00:27:00`
- Duration: `163.33s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.22s |  |
| generation #1 | PASS | 36.68s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 83.93s | best; initial_candidate |
| profiling | PASS | 20.82s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 626 --task-end 626
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
