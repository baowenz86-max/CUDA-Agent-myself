# CUDA Agent Workflow Report

- Task ID: `483`
- Status: **SUCCESS**
- Operators: `["torch.t", "nn.AdaptiveAvgPool1d"]`
- Started: `2026-11-10T17:04:29`
- Finished: `2026-11-10T17:11:44`
- Duration: `435.54s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 58.76s |  |
| generation #1 | PASS | 179.11s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; debug_forward: prohibited torch computation torch.transpose |
| repair API call #1 | PASS | 11.85s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 163.22s | best; correctness_success |
| profiling | PASS | 20.68s |  |
| artifacts | PASS | 1.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 483 --task-end 483
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
