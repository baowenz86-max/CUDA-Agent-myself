# CUDA Agent Workflow Report

- Task ID: `817`
- Status: **SUCCESS**
- Operators: `["torch.cos", "nn.Identity"]`
- Started: `2026-11-11T15:17:47`
- Finished: `2026-11-11T15:20:32`
- Duration: `164.40s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.41s |  |
| generation #1 | PASS | 38.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.03s | best; initial_candidate; forward: prohibited torch computation torch.cos
debug_forward: prohibited torch computation torch.cos |
| repair API call #1 | PASS | 8.15s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 53.94s | best; correctness_success |
| profiling | PASS | 25.56s |  |
| artifacts | PASS | 25.86s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 817 --task-end 817
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
