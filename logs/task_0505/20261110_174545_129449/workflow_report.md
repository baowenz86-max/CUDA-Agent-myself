# CUDA Agent Workflow Report

- Task ID: `505`
- Status: **SUCCESS**
- Operators: `["torch.diag", "nn.GroupNorm"]`
- Started: `2026-11-10T17:45:45`
- Finished: `2026-11-10T17:49:13`
- Duration: `208.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.90s |  |
| generation #1 | PASS | 60.63s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 97.75s | best; initial_candidate |
| profiling | PASS | 18.75s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 505 --task-end 505
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
