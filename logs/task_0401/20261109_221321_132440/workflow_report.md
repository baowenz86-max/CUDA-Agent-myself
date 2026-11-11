# CUDA Agent Workflow Report

- Task ID: `401`
- Status: **SUCCESS**
- Operators: `["torch.max", "nn.MaxPool2d"]`
- Started: `2026-11-09T22:13:21`
- Finished: `2026-11-09T22:18:18`
- Duration: `297.21s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.38s |  |
| generation #1 | PASS | 79.20s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 99.97s | best; initial_candidate |
| profiling | PASS | 93.75s |  |
| artifacts | PASS | 1.87s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 401 --task-end 401
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
