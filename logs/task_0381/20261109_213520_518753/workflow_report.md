# CUDA Agent Workflow Report

- Task ID: `381`
- Status: **SUCCESS**
- Operators: `["F.hardtanh", "nn.InstanceNorm3d"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:43:00`
- Duration: `459.60s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 74.93s |  |
| generation #1 | PASS | 96.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 83.50s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0381/kernels/custom_ops.cu(184): error: calling a __device__ function("ceil_div(int, int)") from a __host__ function("custom_hardtanh_launcher") is not allowed
      int blocks = (int)ceil_div(numel, threads);
                        ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0381/kernels/custom_ops.cu(197): error: calling a __device__ function("_Z8ceil_div1?1?") from a __host__ function("custom_slice_last_dim_launcher") is not allowed
      int blocks = (int)ceil_div(output_numel, threads);
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0381/kernels/custom_ops.cu(230): error: identifier "instance_norm_fused_kernel" is undefined
      instance_norm_fused_kernel<<<grid, block, shared_mem_size, stream>>>(
      ^ |
| repair API call #1 | PASS | 19.08s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 150.06s | best; correctness_success |
| profiling | PASS | 33.82s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 381 --task-end 381
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
