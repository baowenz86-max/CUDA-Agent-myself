# CUDA Agent Workflow Report

- Task ID: `625`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.clip"]`
- Started: `2026-11-11T00:18:34`
- Finished: `2026-11-11T00:24:16`
- Duration: `342.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.86s |  |
| generation #1 | PASS | 51.89s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 147.27s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 66796335 / 134217728 (49.8%)
Greatest absolute difference: 1.0 at index (0, 0, 0, 1) (up to 0.01 allowed)
Mismatched elements: 66790390 / 134217728 (49.8%)
Mismatched elements: 66792547 / 134217728 (49.8%)
Greatest absolute difference: 1.0 at index (0, 0, 0, 0) (up to 0.01 allowed)
Mismatched elements: 66792747 / 134217728 (49.8%)
Mismatched elements: 66799141 / 134217728 (49.8%)
Greatest absolute difference: 1.0 at index (0, 0, 0, 3) (up to 0.01 allowed) |
| repair API call #1 | PASS | 24.36s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 23.60s | rejected; missing_normal_observations; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
Compile cache miss: incremental Ninja build
Compile success: cuda_extension.so
[TIME] Compilation took 3.33s

[PASSED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0625/20261111_001834_324830/correctness/interface.json

 |
| repair API call #2 | PASS | 19.32s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 45.77s | best; correctness_success |
| profiling | PASS | 12.27s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 625 --task-end 625
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
