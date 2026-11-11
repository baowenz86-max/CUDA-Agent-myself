# CUDA Agent Workflow Report

- Task ID: `723`
- Status: **SUCCESS**
- Operators: `["torch.arange", "nn.AvgPool2d"]`
- Started: `2026-11-11T12:49:25`
- Finished: `2026-11-11T12:52:29`
- Duration: `183.81s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.43s |  |
| generation #1 | PASS | 36.64s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 123.60s | best; initial_candidate |
| profiling | PASS | 15.39s |  |
| artifacts | PASS | 1.71s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 723 --task-end 723
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
