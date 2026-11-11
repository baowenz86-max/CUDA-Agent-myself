# CUDA Agent Workflow Report

- Task ID: `370`
- Status: **SUCCESS**
- Operators: `["torch.equal", "nn.GroupNorm"]`
- Started: `2026-11-09T14:59:56`
- Finished: `2026-11-09T15:06:16`
- Duration: `379.70s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.47s |  |
| generation #1 | PASS | 93.23s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; forward: prohibited torch computation torch.equal
debug_forward: prohibited torch computation torch.equal |
| repair API call #1 | PASS | 11.91s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 202.31s | best; correctness_success |
| profiling | PASS | 43.07s |  |
| artifacts | PASS | 1.58s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 370 --task-end 370
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
