# CUDA Agent Workflow Report

- Task ID: `665`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.MaxPool2d"]`
- Started: `2026-11-11T10:14:24`
- Finished: `2026-11-11T10:17:37`
- Duration: `193.00s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.05s |  |
| generation #1 | PASS | 45.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 67.66s | best; initial_candidate |
| profiling | PASS | 63.81s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 665 --task-end 665
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
