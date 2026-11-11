# CUDA Agent Workflow Report

- Task ID: `744`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.BatchNorm2d"]`
- Started: `2026-11-11T13:19:29`
- Finished: `2026-11-11T13:22:28`
- Duration: `179.73s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.82s |  |
| generation #1 | PASS | 70.09s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 58.97s | best; initial_candidate |
| profiling | PASS | 23.15s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 744 --task-end 744
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
