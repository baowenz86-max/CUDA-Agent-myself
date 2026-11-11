# CUDA Agent Workflow Report

- Task ID: `664`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.is_complex"]`
- Started: `2026-11-11T10:14:24`
- Finished: `2026-11-11T10:18:54`
- Duration: `269.94s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 26.61s |  |
| generation #1 | PASS | 32.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.08s | best; initial_candidate; debug_forward: prohibited torch computation torch.is_complex
debug_forward: torch.ops.cuda_extension.mean_dim_1 is incompatible with the pybind cuda_extension module |
| repair API call #1 | PASS | 7.01s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 160.09s | best; correctness_success |
| profiling | PASS | 41.57s |  |
| artifacts | PASS | 1.55s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 664 --task-end 664
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
