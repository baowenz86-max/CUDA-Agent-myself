# CUDA Agent Workflow Report

- Task ID: `481`
- Status: **FAILED**
- Operators: `["nn.MultiheadAttention", "nn.GLU", "nn.Linear"]`
- Started: `2026-11-10T17:04:29`
- Finished: `2026-11-10T17:13:36`
- Duration: `547.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 67.99s |  |
| generation #1 | PASS | 318.87s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 30.59s | best; initial_candidate; ValueError: state_dict keys differ from the reference

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0481/20261110_170429_061830/correctness/interface_precompile.json |
| repair API call #1 | PASS | 21.80s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 25.40s | rejected; pipeline_regression; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(154): error: expression must have pointer-to-object type but it has type "int"
                  q_tile[q_local * head_dim + d] = q[(q_global * (batch * num_heads) + bh) * head_dim + d];
                         ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(197): error: expression must have pointer-to-object type but it has type "int"
                              score += q_tile[q_local * head_dim + d] * K_tile[kk * head_dim + d];
                                              ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(310): error: expression must have pointer-to-object type but it has type "int"
                  q_tile[tid * head_dim + d] = q[(q_idx * (batch * num_heads) + bh) * head_dim + d];
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(350): error: expression must have pointer-to-object type but it has type "int"
                          score += q_tile[tid * head_dim + d] * K_tile[kk * head_dim + d];
                                          ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(391): error: identifier "valid_kv" is undefined
                  if (valid_kv) {
                      ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(574): error: expression must have pointer-to-object type but it has type "int"
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(609): error: expression must have pointer-to-object type but it has type "int" |
| repair API call #2 | PASS | 13.89s | Code repairs used: 2/3; base candidate #1 |
| repair #2: compile candidate #3 | FAIL | 5.67s | rejected; pipeline_regression; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(154): error: expression must have pointer-to-object type but it has type "int"
                  q_tile[q_local * head_dim + d] = q[(q_global * (batch * num_heads) + bh) * head_dim + d];
                         ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(197): error: expression must have pointer-to-object type but it has type "int"
                              score += q_tile[q_local * head_dim + d] * K_tile[kk * head_dim + d];
                                              ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(310): error: expression must have pointer-to-object type but it has type "int"
                  q_tile[tid * head_dim + d] = q[(q_idx * (batch * num_heads) + bh) * head_dim + d];
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(350): error: expression must have pointer-to-object type but it has type "int"
                          score += q_tile[tid * head_dim + d] * K_tile[kk * head_dim + d];
                                          ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(391): error: identifier "valid_kv" is undefined
                  if (valid_kv) {
                      ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(574): error: expression must have pointer-to-object type but it has type "int"
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(609): error: expression must have pointer-to-object type but it has type "int" |
| repair API call #3 | PASS | 57.72s | Code repairs used: 3/3; base candidate #1 |
| repair #3: compile candidate #4 | FAIL | 5.50s | rejected; pipeline_regression; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(76): error: expression must have pointer-to-object type but it has type "int"
                  q_tile[tid * head_dim + d] = q[(q_idx * (batch * num_heads) + bh) * head_dim + d];
                         ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/kernels/custom_ops.cu(110): error: expression must have pointer-to-object type but it has type "int"
                          score += q_tile[tid * head_dim + d] * K_tile[kk * head_dim + d];
                                          ^ |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage interface: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0481/utils/preflight.py", line 46, in main
    raise ValueError('\n'.join(errors))
ValueError: state_dict keys differ from the reference

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0481/20261110_170429_061830/correctness/interface_precompile.json


## Reproduction

```bash
python main.py --task-start 481 --task-end 481
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
