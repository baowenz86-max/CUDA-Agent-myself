# CUDA Agent Workflow Report

- Task ID: `663`
- Status: **FAILED**
- Operators: `["torch.sum", "nn.ReLU6"]`
- Started: `2026-11-11T10:14:24`
- Finished: `2026-11-11T10:20:57`
- Duration: `393.08s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.60s |  |
| generation #1 | PASS | 77.07s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 95.45s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0663/kernels/custom_ops.cu(37): error: no instance of function template "compute_offset" matches the argument list
            argument types are: (int64_t, const int64_t *__restrict__, const int64_t *__restrict__, int)
          int64_t in_offset = compute_offset(idx, dims, input_strides, ndim);
                              ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0663/kernels/custom_ops.cu(38): error: no instance of function template "compute_offset" matches the argument list
          int64_t out_offset = compute_offset(idx, dims, output_strides, ndim);
                               ^ |
| repair API call #1 | PASS | 20.66s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 108.65s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 24.99s | Code repairs used: 2/3; base candidate #2 |
| repair #2: resource candidate #3 | FAIL | 46.37s | rejected; terminal_failure; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.00 GiB. GPU 0 has a total capacity of 79.25 GiB of which 20.95 GiB is free. Including non-PyTorch memory, this process has 58.29 GiB memory in use. Of the allocated memory 57.03 GiB is allocated by PyTorch, and 800.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

The above exception was the direct cause of the following exception:
RuntimeError: Comparing
TensorLikePair(
    id=(),
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.00 GiB. GPU 0 has a total capacity of 79.25 GiB of which 22.12 GiB is free. Including non-PyTorch memory, this process has 57.12 GiB memory in use. Of the allocated memory 32.03 GiB is allocated by PyTorch, and 24.61 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |

## Failure Summary

Task terminated: resource; torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.00 GiB. GPU 0 has a total capacity of 79.25 GiB of which 20.95 GiB is free. Including non-PyTorch memory, this process has 58.29 GiB memory in use. Of the allocated memory 57.03 GiB is allocated by PyTorch, and 800.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

The above exception was the direct cause of the following exception:
RuntimeError: Comparing
TensorLikePair(
    id=(),
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 25.00 GiB. GPU 0 has a total capacity of 79.25 GiB of which 22.12 GiB is free. Including non-PyTorch memory, this process has 57.12 GiB memory in use. Of the allocated memory 32.03 GiB is allocated by PyTorch, and 24.61 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

## Reproduction

```bash
python main.py --task-start 663 --task-end 663
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
