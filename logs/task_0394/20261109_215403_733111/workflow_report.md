# CUDA Agent Workflow Report

- Task ID: `394`
- Status: **SUCCESS**
- Operators: `["torch.softmax", "add"]`
- Started: `2026-11-09T21:54:03`
- Finished: `2026-11-09T21:59:25`
- Duration: `322.09s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.85s |  |
| generation #1 | PASS | 83.45s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 182.33s | best; initial_candidate |
| profiling | PASS | 29.95s |  |
| artifacts | PASS | 8.43s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 394 --task-end 394
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
