# CUDA Agent Workflow Report

- Task ID: `674`
- Status: **SUCCESS**
- Operators: `["torch.arange", "nn.BatchNorm2d"]`
- Started: `2026-11-11T10:31:37`
- Finished: `2026-11-11T10:34:20`
- Duration: `162.20s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.45s |  |
| generation #1 | PASS | 51.29s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 58.93s | best; initial_candidate |
| profiling | PASS | 39.09s |  |
| artifacts | PASS | 10.38s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 674 --task-end 674
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
