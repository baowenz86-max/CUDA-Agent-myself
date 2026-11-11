# CUDA Agent Workflow Report

- Task ID: `746`
- Status: **SUCCESS**
- Operators: `["F.softmax", "torch.rot90"]`
- Started: `2026-11-11T13:26:35`
- Finished: `2026-11-11T13:29:33`
- Duration: `177.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.37s |  |
| generation #1 | PASS | 56.79s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 76.12s | best; initial_candidate |
| profiling | PASS | 28.83s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 746 --task-end 746
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
