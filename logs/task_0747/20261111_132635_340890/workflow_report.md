# CUDA Agent Workflow Report

- Task ID: `747`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.transpose"]`
- Started: `2026-11-11T13:26:35`
- Finished: `2026-11-11T13:32:01`
- Duration: `326.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.28s |  |
| generation #1 | PASS | 117.45s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 61.85s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0747/kernels/custom_ops.cu(184): error: function "mean_only_launcher" has already been defined (previous definition at line 96)
  extern "C" void mean_only_launcher(
                  ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0747/kernels/custom_ops.cu(357): error: identifier "transpose3d_kernel" is undefined
          transpose3d_kernel<<<grid_size, block_size, 0, stream>>>(
          ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0747/kernels/custom_ops.cu(369): error: identifier "transpose4d_kernel" is undefined
          transpose4d_kernel<<<grid_size, block_size, 0, stream>>>( |
| repair API call #1 | PASS | 30.30s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 41.62s | best; correctness_success |
| profiling | PASS | 28.76s |  |
| artifacts | PASS | 20.77s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 747 --task-end 747
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
