# CUDA Agent Workflow Report

- Task ID: `694`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.Sigmoid"]`
- Started: `2026-11-11T11:00:43`
- Finished: `2026-11-11T11:04:39`
- Duration: `236.75s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 48.82s |  |
| generation #1 | PASS | 80.65s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 83.40s | best; initial_candidate |
| profiling | PASS | 21.92s |  |
| artifacts | PASS | 1.89s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 694 --task-end 694
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
