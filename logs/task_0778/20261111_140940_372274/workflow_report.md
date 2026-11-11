# CUDA Agent Workflow Report

- Task ID: `778`
- Status: **SUCCESS**
- Operators: `["torch.isin", "nn.MaxPool1d"]`
- Started: `2026-11-11T14:09:40`
- Finished: `2026-11-11T14:16:50`
- Duration: `429.85s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.84s |  |
| generation #1 | PASS | 90.57s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 68.52s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0778/kernels/custom_ops.cu(227): error: more than one instance of overloaded function "isin_custom_launcher" has "C" linkage
  extern "C" void isin_custom_launcher(
                  ^
 |
| repair API call #1 | PASS | 19.31s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 77.62s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 63.53s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 94.46s | best; correctness_success |
| profiling | PASS | 10.97s |  |
| artifacts | PASS | 1.75s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 778 --task-end 778
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
