# CUDA Agent Workflow Report

- Task ID: `820`
- Status: **SUCCESS**
- Operators: `["torch.ceil", "nn.Upsample"]`
- Started: `2026-11-11T15:17:47`
- Finished: `2026-11-11T15:22:35`
- Duration: `287.40s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.22s |  |
| generation #1 | PASS | 85.77s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 50.63s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0820/kernels/custom_ops.cu(155): error: identifier "ceil_tensor_kernel_grid_stride" is undefined
          ceil_tensor_kernel_grid_stride<<<grid, threads, 0, stream>>>(input, output, numel);
          ^
 |
| repair API call #1 | PASS | 22.28s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 98.45s | best; correctness_success |
| profiling | PASS | 22.36s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 820 --task-end 820
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
