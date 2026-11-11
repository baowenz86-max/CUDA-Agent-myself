# CUDA Agent Workflow Report

- Task ID: `415`
- Status: **SUCCESS**
- Operators: `["nn.GroupNorm"]`
- Started: `2026-11-09T22:27:13`
- Finished: `2026-11-09T22:30:57`
- Duration: `224.00s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.21s |  |
| generation #1 | PASS | 79.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 111.43s | best; initial_candidate |
| profiling | PASS | 18.31s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 415 --task-end 415
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
