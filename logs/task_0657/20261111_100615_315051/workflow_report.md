# CUDA Agent Workflow Report

- Task ID: `657`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.mean"]`
- Started: `2026-11-11T10:06:15`
- Finished: `2026-11-11T10:11:35`
- Duration: `320.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 57.37s |  |
| generation #1 | PASS | 37.88s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 148.85s | best; initial_candidate |
| profiling | PASS | 74.14s |  |
| artifacts | PASS | 1.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 657 --task-end 657
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
