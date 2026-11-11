# CUDA Agent Workflow Report

- Task ID: `590`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.clone"]`
- Started: `2026-11-10T20:36:03`
- Finished: `2026-11-10T20:39:34`
- Duration: `210.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.82s |  |
| generation #1 | PASS | 67.11s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 114.44s | best; initial_candidate |
| profiling | PASS | 14.51s |  |
| artifacts | PASS | 1.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 590 --task-end 590
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
