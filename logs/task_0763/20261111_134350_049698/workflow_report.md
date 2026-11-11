# CUDA Agent Workflow Report

- Task ID: `763`
- Status: **SUCCESS**
- Operators: `["torch.sum", "F.relu"]`
- Started: `2026-11-11T13:43:50`
- Finished: `2026-11-11T13:47:05`
- Duration: `195.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.07s |  |
| generation #1 | PASS | 38.13s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 116.11s | best; initial_candidate |
| profiling | PASS | 19.63s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 763 --task-end 763
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
