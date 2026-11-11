# CUDA Agent Workflow Report

- Task ID: `149`
- Status: **FAILED**
- Operators: `["torch.view_as_complex", "nn.MultiheadAttention"]`
- Started: `2026-11-06T17:11:54`
- Finished: `2026-11-06T17:29:02`
- Duration: `1028.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 41.46s |  |
| generation #1 | PASS | 745.15s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 19.13s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/kernels/custom_ops.cu(85): error: identifier "printf" is undefined
     (handle, opA_cublas, opB_cublas, m_cublas, n_cublas, k, &alpha, B, ldb, A, lda, &beta, C, ldc); if (status != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error at %s:%d\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/kernels/custom_ops.cu", 85); return; } } while (0)
                                                                                                                                            ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/kernels/custom_ops.cu(181): error: identifier "printf" is undefined
     (&handle); if (status != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error at %s:%d\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/kernels/custom_ops.cu", 181); return; } } while (0);
                                                       ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/kernels/custom_ops.cu(182): error: identifier "printf" is undefined
     (handle, stream); if (status != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error at %s:%d\n", "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/kernels/custom_ops.cu", 182); return; } } while (0);
                                                              ^ |
| repair API call #1 | PASS | 43.09s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 4.79s | rejected; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/build/forced_compile/cuda_extension.so: undefined symbol: view_as_complex_launcher
[TIME] Compilation took 4.74s |
| repair API call #2 | PASS | 34.37s | Code repairs used: 2/3; base candidate #1 |
| repair #2: compile candidate #3 | FAIL | 4.78s | rejected; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0149/build/forced_compile/cuda_extension.so: undefined symbol: view_as_complex_launcher
[TIME] Compilation took 4.72s |
| repair API call #3 | PASS | 119.09s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 16.01s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 5170122 / 8388608 (61.6%)
Greatest absolute difference: 0.10283736139535904 at index (15, 1356, 252) (up to 0.01 allowed)
Mismatched elements: 2656752 / 8388608 (31.7%)
Greatest absolute difference: 0.05192007124423981 at index (2, 704, 40) (up to 0.01 allowed)
Mismatched elements: 2653615 / 8388608 (31.6%)
Greatest absolute difference: 0.05311056971549988 at index (7, 1633, 165) (up to 0.01 allowed)
Mismatched elements: 2656746 / 8388608 (31.7%)
Greatest absolute difference: 0.058472104370594025 at index (15, 1718, 86) (up to 0.01 allowed)
Mismatched elements: 2612825 / 8388608 (31.1%)
Greatest absolute difference: 0.0490519218146801 at index (5, 334, 46) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [16, 2048, 256], "dtype": "torch.float32", "device": "cuda:0", "stride": [524288, 256, 1], "numel": 8388608}, "expected": {"type": "tensor", "shape": [16, 2048, 256], "dtype": "torch.float32", "device": "cuda:0", "stride": [256, 4096, 1], "numel": 8388608}, "kind": "tensor_values", "mismatched": 5170122, "numel": 8388608, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.6163265705108643, "max_absolute_error": 0.10283735953271389, "max_relative_error": 3174463.316562761, "samples": [{"index": [0, 0, 0], "actual": "-0.020507778972387314", "expected": "0.014057940803468227"}, {"index": [0, 0, 1], "actual": "0.011803125962615013", "expected": "-0.005394637119024992"}, {"index": [0, 0, 2], "actual": "0.039355553686618805", "expected": "-0.006397514138370752"}, {"index": [0, 0, 3], "actual": "-0.018277471885085106", "expected": "0.01768427900969982"}, {"index": [0, 0, 6], "actual": "0.027824141085147858", "expected": "0.009671377949416637"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 5170122 / 8388608 (61.6%)\nGreatest absolute difference: 0.10283736139535904 at index (15, 1356, 252) (up to 0.01 allowed)\nGreatest relative difference: 3174463.5 at index (3, 1528, 193) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 149 --task-end 149
```
