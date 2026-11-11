# CUDA Agent Workflow Report

- Task ID: `714`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.Softplus"]`
- Started: `2026-11-11T11:32:58`
- Finished: `2026-11-11T11:35:32`
- Duration: `153.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.19s |  |
| generation #1 | PASS | 36.83s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 49.85s | best; initial_candidate |
| profiling | PASS | 31.99s |  |
| artifacts | PASS | 30.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 714 --task-end 714
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
