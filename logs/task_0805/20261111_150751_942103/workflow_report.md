# CUDA Agent Workflow Report

- Task ID: `805`
- Status: **FAILED**
- Operators: `["torch.stack", "F.relu"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:17:41`
- Duration: `589.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.41s |  |
| generation #1 | PASS | 120.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 145.27s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0805/kernels/custom_ops.cu(177): error: function "stack_relu_forward_launcher" has already been defined (previous definition at line 68)
  extern "C" void stack_relu_forward_launcher(
                  ^
 |
| repair API call #1 | PASS | 16.22s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 74.36s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 298601932 / 402653184 (74.2%)
Greatest absolute difference: 5.766856670379639 at index (362, 0, 0, 205, 135) (up to 0.01 allowed)
Mismatched elements: 298688596 / 402653184 (74.2%)
Greatest absolute difference: 5.583926200866699 at index (121, 0, 2, 100, 170) (up to 0.01 allowed)
Mismatched elements: 298541508 / 402653184 (74.1%)
Greatest absolute difference: 5.6834845542907715 at index (441, 0, 0, 213, 209) (up to 0.01 allowed)
Mismatched elements: 298621064 / 402653184 (74.2%)
Greatest absolute difference: 5.516506671905518 at index (129, 0, 0, 228, 58) (up to 0.01 allowed)
Mismatched elements: 298604780 / 402653184 (74.2%)
Greatest absolute difference: 5.559072017669678 at index (11, 0, 2, 21, 221) (up to 0.01 allowed) |
| repair API call #2 | PASS | 78.28s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 44.78s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #3 | PASS | 32.45s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 52.99s | rejected; missing_normal_observations; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [512, 4, 3, 256, 256], "dtype": "torch.float32", "device": "cuda:0", "stride": [786432, 196608, 65536, 256, 1], "numel": 402653184}, "expected": {"type": "tensor", "shape": [512, 4, 3, 256, 256], "dtype": "torch.float32", "device": "cuda:0", "stride": [786432, 196608, 65536, 256, 1], "numel": 402653184}, "kind": "tensor_values", "mismatched": 298601932, "numel": 402653184, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.7415859202543894, "max_absolute_error": 5.766856670379639, "max_relative_error": 4.81798505783081e+30, "samples": [{"index": [0, 0, 0, 3, 0], "actual": "0.8537248373031616", "expected": "0.33054420351982117"}, {"index": [0, 0, 0, 3, 1], "actual": "1.5038443803787231", "expected": "0.0"}, {"index": [0, 0, 0, 3, 2], "actual": "0.1817638874053955", "expected": "0.0"}, {"index": [0, 0, 0, 3, 3], "actual": "0.07021576911211014", "expected": "2.9474782943725586"}, {"index": [0, 0, 0, 3, 4], "actual": "1.5879098176956177", "expected": "0.7050694823265076"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 298601932 / 402653184 (74.2%)\nGreatest absolute difference: 5.766856670379639 at index (362, 0, 0, 205, 135) (up to 0.01 allowed)\nGreatest relative difference: inf at index (0, 0, 0, 3, 1) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 805 --task-end 805
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
