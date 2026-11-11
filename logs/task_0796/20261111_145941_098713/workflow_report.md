# CUDA Agent Workflow Report

- Task ID: `796`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.tan"]`
- Started: `2026-11-11T14:59:41`
- Finished: `2026-11-11T15:03:39`
- Duration: `238.58s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.09s |  |
| generation #1 | PASS | 48.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 97.55s | best; initial_candidate |
| profiling | PASS | 52.93s |  |
| artifacts | PASS | 17.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 796 --task-end 796
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
