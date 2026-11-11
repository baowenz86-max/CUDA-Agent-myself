# CUDA Agent Workflow Report

- Task ID: `659`
- Status: **FAILED**
- Operators: `["torch.cat", "F.scaled_dot_product_attention"]`
- Started: `2026-11-11T10:06:15`
- Finished: `2026-11-11T10:14:22`
- Duration: `487.50s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.43s |  |
| generation #1 | PASS | 84.61s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 82.92s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0659/kernels/custom_ops.cu(34): error: identifier "copy_strided_kernel" is undefined
          copy_strided_kernel<<<grid, block, 0, stream>>>(
          ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0659/kernels/custom_ops.cu(78): error: identifier "add_bias_kernel" is undefined
          add_bias_kernel<<<bias_grid, bias_block, 0, stream>>>(
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0659/kernels/custom_ops.cu(117): error: identifier "attention_kernel" is undefined
      attention_kernel<<<grid, block, 0, stream>>>(
      ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0659/kernels/custom_ops.cu(139): error: identifier "concat_kernel" is undefined
      concat_kernel<<<blocks, threads, 0, stream>>>( |
| repair API call #1 | PASS | 28.18s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 106.56s | best; pipeline_progress; RuntimeError: linear_forward kernel launch failed: an illegal memory access was encountered

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0659/20261111_100615_336526/correctness/verification.json |
| repair API call #2 | PASS | 36.63s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 33.15s | rejected; tie_keep_previous; RuntimeError: linear_forward kernel launch failed: an illegal memory access was encountered

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0659/20261111_100615_336526/correctness/verification.json |
| repair API call #3 | PASS | 37.94s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 31.43s | rejected; tie_keep_previous; RuntimeError: setStorage: sizes [16, 1024, 512], strides [512, 524288, 1], storage offset 0, and itemsize 4 requiring a storage size of 2145419264 are out of bounds for storage of size 33554432

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0659/20261111_100615_336526/correctness/verification.json |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"kind": "runtime", "message": "RuntimeError: linear_forward kernel launch failed: an illegal memory access was encountered"}

## Reproduction

```bash
python main.py --task-start 659 --task-end 659
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
