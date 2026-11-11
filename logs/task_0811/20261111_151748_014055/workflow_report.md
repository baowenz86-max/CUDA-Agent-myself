# CUDA Agent Workflow Report

- Task ID: `811`
- Status: **SUCCESS**
- Operators: `["nn.Upsample", "torch.softmax"]`
- Started: `2026-11-11T15:17:48`
- Finished: `2026-11-11T15:32:28`
- Duration: `880.28s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 58.21s |  |
| generation #1 | PASS | 74.07s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 111.76s | best; initial_candidate; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #1 | PASS | 339.08s | Code repairs used: 1/3; base candidate #1 |
| repair #1: sources candidate #2 | FAIL | 0.05s | rejected; missing_normal_observations; forward: cuda_extension.custom_upsample_bilinear has no visible pybind registration
forward: cuda_extension.custom_softmax_dim has no visible pybind registration
debug_forward: cuda_extension.custom_upsample_bilinear has no visible pybind registration
debug_forward: cuda_extension.custom_softmax_dim has no visible pybind registration |
| repair API call #2 | PASS | 52.21s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 46.40s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 191657413 / 201326592 (95.2%)
Greatest absolute difference: 0.9199802875518799 at index (21, 2, 965, 760) (up to 0.01 allowed)
Mismatched elements: 191654791 / 201326592 (95.2%)
Greatest absolute difference: 0.9144642353057861 at index (44, 1, 842, 794) (up to 0.01 allowed)
Mismatched elements: 191659053 / 201326592 (95.2%)
Greatest absolute difference: 0.9187573194503784 at index (31, 0, 960, 691) (up to 0.01 allowed)
Mismatched elements: 191655615 / 201326592 (95.2%)
Greatest absolute difference: 0.9086498618125916 at index (45, 0, 757, 836) (up to 0.01 allowed)
Mismatched elements: 191660762 / 201326592 (95.2%)
Greatest absolute difference: 0.9279559254646301 at index (32, 1, 895, 901) (up to 0.01 allowed) |
| repair API call #3 | PASS | 161.41s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | PASS | 13.49s | best; correctness_success |
| profiling | PASS | 21.52s |  |
| artifacts | PASS | 1.11s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 811 --task-end 811
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
