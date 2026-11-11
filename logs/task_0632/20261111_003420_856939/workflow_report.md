# CUDA Agent Workflow Report

- Task ID: `632`
- Status: **SUCCESS**
- Operators: `["torch.sum", "F.interpolate"]`
- Started: `2026-11-11T00:34:20`
- Finished: `2026-11-11T00:38:07`
- Duration: `226.29s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.29s |  |
| generation #1 | PASS | 50.38s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; sum_reduction: py::arg order/name/default differs |
| repair API call #1 | PASS | 26.08s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 112.44s | best; correctness_success |
| profiling | PASS | 30.26s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 632 --task-end 632
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
