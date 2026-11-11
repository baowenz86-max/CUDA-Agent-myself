# CUDA Agent Workflow Report

- Task ID: `332`
- Status: **FAILED**
- Operators: `["nn.AdaptiveAvgPool1d", "nn.LSTM"]`
- Started: `2026-11-07T16:56:26`
- Finished: `2026-11-07T17:00:59`
- Duration: `273.12s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.77s |  |
| generation #1 | PASS | 119.48s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 30.53s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu(63): error: identifier "printf" is undefined
    do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu", 63, cudaGetErrorString(err)); return; } } while (0);
                                                                         ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu(110): error: identifier "printf" is undefined
    do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu", 110, cudaGetErrorString(err)); return; } } while (0);
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu(393): error: expected a ";"
                void combine_bias_kernel(float* out, const float* a, const float* b, int n) {
                                                                                            ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu(415): error: identifier "printf" is undefined
    do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu", 415, cudaGetErrorString(err)); return; } } while (0);
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu(499): error: identifier "printf" is undefined
    do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu", 499, cudaGetErrorString(err)); return; } } while (0);
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu(544): error: identifier "printf" is undefined
    do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0332/kernels/custom_ops.cu", 544, cudaGetErrorString(err)); return; } } while (0); |
| repair API call #1 | PASS | 35.63s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 6.45s | best; ValueError: linear: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'weight: torch.Tensor', 'bias: torch.Tensor']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0332/20261107_165626_712185/correctness/interface.json |
| repair API call #2 | PASS | 30.42s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 24.26s | best; AttributeError: 'ModelNew' object has no attribute 'pooling_length'

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0332/20261107_165626_712185/correctness/verification.json |
| repair API call #3 | PASS | 8.73s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 10.59s | rejected; AttributeError: 'ModelNew' object has no attribute 'hidden_size'

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0332/20261107_165626_712185/correctness/verification.json |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 3, stage verification: {"kind": "runtime", "message": "AttributeError: 'ModelNew' object has no attribute 'pooling_length'"}

## Reproduction

```bash
python main.py --task-start 332 --task-end 332
```
