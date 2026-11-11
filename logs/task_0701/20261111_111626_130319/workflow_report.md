# CUDA Agent Workflow Report

- Task ID: `701`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.split"]`
- Started: `2026-11-11T11:16:26`
- Finished: `2026-11-11T11:19:49`
- Duration: `203.13s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.10s |  |
| generation #1 | PASS | 63.77s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 55.06s | best; initial_candidate |
| profiling | PASS | 65.48s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 701 --task-end 701
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
