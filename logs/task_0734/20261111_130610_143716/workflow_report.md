# CUDA Agent Workflow Report

- Task ID: `734`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.squeeze"]`
- Started: `2026-11-11T13:06:10`
- Finished: `2026-11-11T13:11:00`
- Duration: `290.55s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.21s |  |
| generation #1 | PASS | 40.74s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 116.03s | best; initial_candidate |
| profiling | PASS | 107.42s |  |
| artifacts | PASS | 2.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 734 --task-end 734
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
