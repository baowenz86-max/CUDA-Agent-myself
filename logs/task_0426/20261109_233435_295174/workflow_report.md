# CUDA Agent Workflow Report

- Task ID: `426`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.flip"]`
- Started: `2026-11-09T23:34:35`
- Finished: `2026-11-09T23:37:27`
- Duration: `171.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.52s |  |
| generation #1 | PASS | 49.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 82.82s | best; initial_candidate |
| profiling | PASS | 14.47s |  |
| artifacts | PASS | 1.16s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 426 --task-end 426
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
