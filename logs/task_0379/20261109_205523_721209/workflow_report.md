# CUDA Agent Workflow Report

- Task ID: `379`
- Status: **SUCCESS**
- Operators: `["torch.max", "torch.exp"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:03:22`
- Duration: `479.22s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 36.29s |  |
| generation #1 | PASS | 66.54s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; forward: torch.ops.cuda_extension.custom_max_dim1 is incompatible with the pybind cuda_extension module
forward: torch.ops.cuda_extension.custom_exp is incompatible with the pybind cuda_extension module
debug_forward: torch.ops.cuda_extension.custom_max_dim1 is incompatible with the pybind cuda_extension module
debug_forward: torch.ops.cuda_extension.custom_exp is incompatible with the pybind cuda_extension module |
| repair API call #1 | PASS | 10.39s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 199.13s | best; correctness_success |
| profiling | PASS | 165.11s |  |
| artifacts | PASS | 1.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 379 --task-end 379
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
