# CUDA Agent Workflow Report

- Task ID: `412`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm2d", "nn.AvgPool2d"]`
- Started: `2026-11-09T22:58:51`
- Finished: `2026-11-09T23:01:19`
- Duration: `147.96s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.64s |  |
| generation #1 | PASS | 78.08s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 30.17s | best; initial_candidate |
| profiling | PASS | 23.95s |  |
| artifacts | PASS | 7.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 412 --task-end 412
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
