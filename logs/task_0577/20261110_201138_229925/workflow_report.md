# CUDA Agent Workflow Report

- Task ID: `577`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.LayerNorm"]`
- Started: `2026-11-10T20:11:38`
- Finished: `2026-11-10T20:22:01`
- Duration: `623.65s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.22s |  |
| generation #1 | PASS | 71.03s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 22.81s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0577/kernels/custom_ops.cu(59): error: expression must have class type but it has type "int"
      int blockDim = blockDim.x;
                     ^
 |
| repair API call #1 | PASS | 21.97s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 36.09s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 329.39s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 37.86s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 67518 / 138240000 (0.0%)
Greatest absolute difference: 810.2372436523438 at index (455, 0, 148, 239) (up to 0.01 allowed)
Mismatched elements: 67514 / 138240000 (0.0%)
Greatest absolute difference: 897.7252197265625 at index (419, 2, 232, 1) (up to 0.01 allowed) |
| repair API call #3 | PASS | 48.10s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | PASS | 38.88s | best; correctness_success |
| profiling | PASS | 10.95s |  |
| artifacts | PASS | 1.17s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 577 --task-end 577
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
