# CUDA Agent Workflow Report

- Task ID: `419`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.log_softmax"]`
- Started: `2026-11-09T23:23:34`
- Finished: `2026-11-09T23:28:52`
- Duration: `318.02s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.94s |  |
| generation #1 | PASS | 64.92s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 129.02s | best; initial_candidate |
| profiling | PASS | 96.00s |  |
| artifacts | PASS | 1.93s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 419 --task-end 419
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
