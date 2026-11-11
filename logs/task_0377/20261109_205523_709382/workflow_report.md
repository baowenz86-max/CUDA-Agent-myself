# CUDA Agent Workflow Report

- Task ID: `377`
- Status: **SUCCESS**
- Operators: `["torch.log_softmax", "nn.MaxPool3d"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:02:30`
- Duration: `426.66s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 26.05s |  |
| generation #1 | PASS | 87.25s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 150.70s | best; initial_candidate |
| profiling | PASS | 160.92s |  |
| artifacts | PASS | 1.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 377 --task-end 377
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
