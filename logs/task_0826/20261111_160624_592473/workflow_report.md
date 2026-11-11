# CUDA Agent Workflow Report

- Task ID: `826`
- Status: **SUCCESS**
- Operators: `["torch.clone", "torch.rot90"]`
- Started: `2026-11-11T16:06:24`
- Finished: `2026-11-11T16:09:15`
- Duration: `170.62s`
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
| generation #1 | PASS | 87.57s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 51.17s | best; initial_candidate |
| profiling | PASS | 22.51s |  |
| artifacts | PASS | 1.28s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 826 --task-end 826
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
