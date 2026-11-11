# CUDA Agent Workflow Report

- Task ID: `441`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm2d", "torch.einsum"]`
- Started: `2026-11-09T23:51:07`
- Finished: `2026-11-09T23:55:45`
- Duration: `277.56s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.83s |  |
| generation #1 | PASS | 70.97s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 159.13s | best; initial_candidate |
| profiling | PASS | 23.90s |  |
| artifacts | PASS | 1.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 441 --task-end 441
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
