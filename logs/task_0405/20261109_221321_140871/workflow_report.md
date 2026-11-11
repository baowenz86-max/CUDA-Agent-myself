# CUDA Agent Workflow Report

- Task ID: `405`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.softmax"]`
- Started: `2026-11-09T22:13:21`
- Finished: `2026-11-09T22:16:55`
- Duration: `214.61s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.14s |  |
| generation #1 | PASS | 58.67s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 101.28s | best; initial_candidate |
| profiling | PASS | 27.79s |  |
| artifacts | PASS | 1.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 405 --task-end 405
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
