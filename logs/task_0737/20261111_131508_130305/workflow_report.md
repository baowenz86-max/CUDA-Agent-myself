# CUDA Agent Workflow Report

- Task ID: `737`
- Status: **SUCCESS**
- Operators: `["torch.arange", "torch.relu"]`
- Started: `2026-11-11T13:15:08`
- Finished: `2026-11-11T13:18:10`
- Duration: `182.41s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 53.04s |  |
| generation #1 | PASS | 23.43s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 79.03s | best; initial_candidate |
| profiling | PASS | 25.38s |  |
| artifacts | PASS | 1.49s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 737 --task-end 737
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
