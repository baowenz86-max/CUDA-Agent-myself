# CUDA Agent Workflow Report

- Task ID: `792`
- Status: **SUCCESS**
- Operators: `["nn.ReLU", "torch.abs"]`
- Started: `2026-11-11T14:55:31`
- Finished: `2026-11-11T14:58:42`
- Duration: `191.18s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.90s |  |
| generation #1 | PASS | 37.57s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 54.34s | best; initial_candidate |
| profiling | PASS | 24.80s |  |
| artifacts | PASS | 55.54s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 792 --task-end 792
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
