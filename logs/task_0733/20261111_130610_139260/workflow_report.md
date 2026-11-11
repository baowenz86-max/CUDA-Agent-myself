# CUDA Agent Workflow Report

- Task ID: `733`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.fmod"]`
- Started: `2026-11-11T13:06:10`
- Finished: `2026-11-11T13:11:31`
- Duration: `321.78s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 33.59s |  |
| generation #1 | PASS | 36.00s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 161.78s | best; initial_candidate |
| profiling | PASS | 88.64s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 733 --task-end 733
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
