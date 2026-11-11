# CUDA Agent Workflow Report

- Task ID: `706`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.AvgPool2d"]`
- Started: `2026-11-11T11:24:24`
- Finished: `2026-11-11T11:26:58`
- Duration: `153.50s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.20s |  |
| generation #1 | PASS | 72.72s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 42.66s | best; initial_candidate |
| profiling | PASS | 12.83s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 706 --task-end 706
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
