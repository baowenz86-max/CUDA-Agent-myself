# CUDA Agent Workflow Report

- Task ID: `413`
- Status: **SUCCESS**
- Operators: `["torch.max", "nn.SyncBatchNorm"]`
- Started: `2026-11-09T22:27:13`
- Finished: `2026-11-09T22:30:48`
- Duration: `214.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.90s |  |
| generation #1 | PASS | 74.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 69.00s | best; initial_candidate |
| profiling | PASS | 21.61s |  |
| artifacts | PASS | 43.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 413 --task-end 413
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
