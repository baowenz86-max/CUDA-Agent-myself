# CUDA Agent Workflow Report

- Task ID: `803`
- Status: **SUCCESS**
- Operators: `["torch.isin", "torch.is_complex"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:12:47`
- Duration: `295.71s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 62.42s |  |
| generation #1 | PASS | 49.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 39.90s | best; initial_candidate |
| profiling | PASS | 20.57s |  |
| artifacts | PASS | 123.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 803 --task-end 803
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
