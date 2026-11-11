# CUDA Agent Workflow Report

- Task ID: `552`
- Status: **SUCCESS**
- Operators: `["F.elu", "torch.abs"]`
- Started: `2026-11-10T19:33:27`
- Finished: `2026-11-10T19:36:58`
- Duration: `210.64s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.39s |  |
| generation #1 | PASS | 41.46s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 71.16s | best; initial_candidate |
| profiling | PASS | 44.52s |  |
| artifacts | PASS | 39.05s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 552 --task-end 552
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
