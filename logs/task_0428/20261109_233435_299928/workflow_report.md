# CUDA Agent Workflow Report

- Task ID: `428`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.div"]`
- Started: `2026-11-09T23:34:35`
- Finished: `2026-11-09T23:36:48`
- Duration: `133.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.98s |  |
| generation #1 | PASS | 36.40s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 53.22s | best; initial_candidate |
| profiling | PASS | 14.18s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 428 --task-end 428
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
