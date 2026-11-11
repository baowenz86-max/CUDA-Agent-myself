# CUDA Agent Workflow Report

- Task ID: `799`
- Status: **SUCCESS**
- Operators: `["torch.stack", "nn.MaxPool2d"]`
- Started: `2026-11-11T14:59:41`
- Finished: `2026-11-11T15:04:18`
- Duration: `277.41s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.87s |  |
| generation #1 | PASS | 74.86s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 151.52s | best; initial_candidate |
| profiling | PASS | 18.61s |  |
| artifacts | PASS | 6.50s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 799 --task-end 799
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
