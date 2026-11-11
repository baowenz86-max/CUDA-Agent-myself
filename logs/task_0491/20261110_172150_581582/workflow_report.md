# CUDA Agent Workflow Report

- Task ID: `491`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm2d", "torch.clamp"]`
- Started: `2026-11-10T17:21:50`
- Finished: `2026-11-10T17:24:22`
- Duration: `151.51s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.97s |  |
| generation #1 | PASS | 52.20s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 37.01s | best; initial_candidate |
| profiling | PASS | 14.73s |  |
| artifacts | PASS | 27.55s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 491 --task-end 491
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
