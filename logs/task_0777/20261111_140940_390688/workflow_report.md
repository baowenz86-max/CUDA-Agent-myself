# CUDA Agent Workflow Report

- Task ID: `777`
- Status: **FAILED**
- Operators: `["torch.mean", "torch.softmax"]`
- Started: `2026-11-11T14:09:40`
- Finished: `2026-11-11T14:20:15`
- Duration: `635.27s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.48s |  |
| generation #1 | PASS | 68.74s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 85.54s | best; initial_candidate; RuntimeError: custom_mean kernel launch failed

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0777/20261111_140940_390688/correctness/verification.json |
| repair API call #1 | PASS | 33.31s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 195.28s | rejected; tie_keep_previous; RuntimeError: custom_mean kernel launch failed

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0777/20261111_140940_390688/correctness/verification.json |
| repair API call #2 | PASS | 36.54s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 74.05s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #3 | PASS | 37.14s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 69.59s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"kind": "runtime", "message": "RuntimeError: custom_mean kernel launch failed"}

## Reproduction

```bash
python main.py --task-start 777 --task-end 777
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
