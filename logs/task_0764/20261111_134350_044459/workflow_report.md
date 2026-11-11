# CUDA Agent Workflow Report

- Task ID: `764`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.softmax"]`
- Started: `2026-11-11T13:43:50`
- Finished: `2026-11-11T13:48:52`
- Duration: `302.68s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.79s |  |
| generation #1 | PASS | 53.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 182.06s | best; initial_candidate |
| profiling | PASS | 51.48s |  |
| artifacts | PASS | 1.11s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 764 --task-end 764
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
