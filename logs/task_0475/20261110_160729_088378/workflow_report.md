# CUDA Agent Workflow Report

- Task ID: `475`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "F.elu"]`
- Started: `2026-11-10T16:07:29`
- Finished: `2026-11-10T16:15:38`
- Duration: `489.52s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.59s |  |
| generation #1 | PASS | 77.59s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 23.60s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0475/build/forced_compile/cuda_extension.so: undefined symbol: custom_elu_launcher
[TIME] Compilation took 20.31s |
| repair API call #1 | PASS | 25.26s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 145.31s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 136.50s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 51.03s | best; correctness_success |
| profiling | PASS | 15.11s |  |
| artifacts | PASS | 1.58s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 475 --task-end 475
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
