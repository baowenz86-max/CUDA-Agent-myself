# CUDA Agent Workflow Report

- Task ID: `380`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.leaky_relu"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:02:16`
- Duration: `412.80s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.09s |  |
| generation #1 | PASS | 70.28s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 138.34s | best; initial_candidate |
| profiling | PASS | 184.45s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 380 --task-end 380
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
