# CUDA Agent Workflow Report

- Task ID: `798`
- Status: **SUCCESS**
- Operators: `["torch.isin", "nn.SyncBatchNorm"]`
- Started: `2026-11-11T14:59:41`
- Finished: `2026-11-11T15:03:43`
- Duration: `242.90s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.96s |  |
| generation #1 | PASS | 69.45s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 153.87s | best; initial_candidate |
| profiling | PASS | 15.09s |  |
| artifacts | PASS | 1.49s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 798 --task-end 798
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
