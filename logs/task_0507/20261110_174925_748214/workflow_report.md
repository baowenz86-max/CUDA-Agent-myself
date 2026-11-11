# CUDA Agent Workflow Report

- Task ID: `507`
- Status: **SUCCESS**
- Operators: `["F.leaky_relu", "nn.AvgPool2d"]`
- Started: `2026-11-10T17:49:25`
- Finished: `2026-11-10T17:52:26`
- Duration: `180.37s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.68s |  |
| generation #1 | PASS | 61.23s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 85.58s | best; initial_candidate |
| profiling | PASS | 14.16s |  |
| artifacts | PASS | 1.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 507 --task-end 507
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
