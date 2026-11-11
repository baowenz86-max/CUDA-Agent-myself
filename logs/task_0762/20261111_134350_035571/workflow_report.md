# CUDA Agent Workflow Report

- Task ID: `762`
- Status: **SUCCESS**
- Operators: `["torch.arange", "F.pad"]`
- Started: `2026-11-11T13:43:50`
- Finished: `2026-11-11T13:46:48`
- Duration: `178.12s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.03s |  |
| generation #1 | PASS | 43.72s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 67.63s | best; initial_candidate |
| profiling | PASS | 22.01s |  |
| artifacts | PASS | 40.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 762 --task-end 762
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
