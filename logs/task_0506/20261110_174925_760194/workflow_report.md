# CUDA Agent Workflow Report

- Task ID: `506`
- Status: **SUCCESS**
- Operators: `["torch.relu", "nn.MaxPool1d"]`
- Started: `2026-11-10T17:49:25`
- Finished: `2026-11-10T17:51:58`
- Duration: `153.21s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.22s |  |
| generation #1 | PASS | 48.64s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 65.85s | best; initial_candidate |
| profiling | PASS | 12.87s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 506 --task-end 506
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
