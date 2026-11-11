# CUDA Agent Workflow Report

- Task ID: `434`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.cumsum"]`
- Started: `2026-11-09T23:41:58`
- Finished: `2026-11-09T23:45:10`
- Duration: `191.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.48s |  |
| generation #1 | PASS | 46.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 70.46s | best; initial_candidate |
| profiling | PASS | 27.29s |  |
| artifacts | PASS | 39.82s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 434 --task-end 434
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
