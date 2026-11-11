# CUDA Agent Workflow Report

- Task ID: `768`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.one_hot"]`
- Started: `2026-11-11T13:50:45`
- Finished: `2026-11-11T13:53:48`
- Duration: `183.15s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.61s |  |
| generation #1 | PASS | 61.65s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 75.12s | best; initial_candidate |
| profiling | PASS | 26.82s |  |
| artifacts | PASS | 11.85s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 768 --task-end 768
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
