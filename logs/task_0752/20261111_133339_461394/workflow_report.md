# CUDA Agent Workflow Report

- Task ID: `752`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.erf"]`
- Started: `2026-11-11T13:33:39`
- Finished: `2026-11-11T13:37:02`
- Duration: `202.92s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.03s |  |
| generation #1 | PASS | 50.70s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 52.82s | best; initial_candidate |
| profiling | PASS | 27.24s |  |
| artifacts | PASS | 64.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 752 --task-end 752
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
