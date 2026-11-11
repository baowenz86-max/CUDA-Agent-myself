# CUDA Agent Workflow Report

- Task ID: `501`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool2d"]`
- Started: `2026-11-10T17:45:45`
- Finished: `2026-11-10T17:48:29`
- Duration: `164.24s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.50s |  |
| generation #1 | PASS | 60.59s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 61.94s | best; initial_candidate |
| profiling | PASS | 16.40s |  |
| artifacts | PASS | 1.75s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 501 --task-end 501
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
