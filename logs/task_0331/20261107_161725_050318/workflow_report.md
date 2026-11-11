# CUDA Agent Workflow Report

- Task ID: `331`
- Status: **FAILED**
- Operators: `["F.gelu", "torch.gt"]`
- Started: `2026-11-07T16:17:25`
- Finished: `2026-11-07T16:25:06`
- Duration: `461.92s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.91s |  |
| generation #1 | PASS | 113.22s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 72.92s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0331/kernels/custom_ops.cu(214): error: function "linear_forward_launcher" has already been defined (previous definition at line 69)
  void linear_forward_launcher(const float* input,
       ^
 |
| repair API call #1 | PASS | 17.84s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 78.21s | rejected; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0331/build/forced_compile/cuda_extension.so: undefined symbol: launch_add
[TIME] Compilation took 4.82s |
| repair API call #2 | PASS | 20.14s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 80.14s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 1536 / 301989888 (0.0%)
Greatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed) |
| repair API call #3 | PASS | 38.83s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 32.29s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 1536 / 301989888 (0.0%)
Greatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [1536, 128, 1536], "dtype": "torch.float32", "device": "cuda:0", "stride": [196608, 1536, 1], "numel": 301989888}, "expected": {"type": "tensor", "shape": [1536, 128, 1536], "dtype": "torch.float32", "device": "cuda:0", "stride": [196608, 1536, 1], "numel": 301989888}, "kind": "tensor_values", "mismatched": 1536, "numel": 301989888, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 5.086263020833333e-06, "max_absolute_error": 1.0000001192092896, "max_relative_error": 1854.6599618075322, "samples": [{"index": [0, 58, 1038], "actual": "0.7070679068565369", "expected": "-0.29293209314346313"}, {"index": [1, 58, 1038], "actual": "1.025524616241455", "expected": "0.0255245603621006"}, {"index": [2, 58, 1038], "actual": "1.608433723449707", "expected": "0.6084337830543518"}, {"index": [3, 58, 1038], "actual": "0.9328385591506958", "expected": "-0.06716146320104599"}, {"index": [4, 58, 1038], "actual": "0.6356709599494934", "expected": "-0.3643290400505066"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 1536 / 301989888 (0.0%)\nGreatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed)\nGreatest relative difference: 1854.659912109375 at index (1274, 58, 1038) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 331 --task-end 331
```
