# CUDA Agent Workflow Report

- Task ID: `765`
- Status: **SUCCESS**
- Operators: `["torch.stack", "torch.bmm"]`
- Started: `2026-11-11T13:43:50`
- Finished: `2026-11-11T13:47:24`
- Duration: `214.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.59s |  |
| generation #1 | PASS | 39.10s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 119.82s | best; initial_candidate |
| profiling | PASS | 31.08s |  |
| artifacts | PASS | 1.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 765 --task-end 765
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
