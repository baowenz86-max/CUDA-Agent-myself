# CUDA Agent Workflow Report

- Task ID: `700`
- Status: **FAILED**
- Operators: `["torch.cat", "nn.Unfold"]`
- Started: `2026-11-11T11:06:54`
- Finished: `2026-11-11T11:16:24`
- Duration: `570.55s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.58s |  |
| generation #1 | PASS | 121.98s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.06s | best; initial_candidate; unfold_forward.kernel_size: C++ type/nullable differs
unfold_forward.dilation: C++ type/nullable differs
unfold_forward.padding: C++ type/nullable differs
unfold_forward.stride: C++ type/nullable differs |
| repair API call #1 | PASS | 48.66s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 162.50s | best; pipeline_progress; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #2 | PASS | 89.06s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 44.76s | rejected; tie_keep_previous; torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1 |
| repair API call #3 | PASS | 43.30s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 30.28s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 349488264 / 377487360 (92.6%)
Greatest absolute difference: 7.856401443481445 at index (7, 15, 171734) (up to 0.01 allowed)
Mismatched elements: 349500915 / 377487360 (92.6%)
Greatest absolute difference: 7.899350166320801 at index (5, 0, 18026) (up to 0.01 allowed)
Mismatched elements: 349486540 / 377487360 (92.6%)
Greatest absolute difference: 7.859238624572754 at index (5, 15, 40954) (up to 0.01 allowed)
Mismatched elements: 349491989 / 377487360 (92.6%)
Greatest absolute difference: 7.382865905761719 at index (7, 0, 201130) (up to 0.01 allowed)
Mismatched elements: 349495158 / 377487360 (92.6%)
Greatest absolute difference: 7.796558380126953 at index (10, 0, 261730) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [16, 90, 262144], "dtype": "torch.float32", "device": "cuda:0", "stride": [23592960, 262144, 1], "numel": 377487360}, "expected": {"type": "tensor", "shape": [16, 90, 262144], "dtype": "torch.float32", "device": "cuda:0", "stride": [23592960, 262144, 1], "numel": 377487360}, "kind": "tensor_values", "mismatched": 349488264, "numel": 377487360, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9258277257283528, "max_absolute_error": 7.856401681900024, "max_relative_error": 7196830.924019563, "samples": [{"index": [0, 45, 514], "actual": "-0.31149616837501526", "expected": "-1.220713496208191"}, {"index": [0, 45, 515], "actual": "1.8251233100891113", "expected": "1.5752907991409302"}, {"index": [0, 45, 516], "actual": "0.03983425348997116", "expected": "-0.33120736479759216"}, {"index": [0, 45, 517], "actual": "-0.20424944162368774", "expected": "0.4247836172580719"}, {"index": [0, 45, 518], "actual": "0.4856487810611725", "expected": "0.19069503247737885"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 349488264 / 377487360 (92.6%)\nGreatest absolute difference: 7.856401443481445 at index (7, 15, 171734) (up to 0.01 allowed)\nGreatest relative difference: 7196831.0 at index (11, 60, 243397) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 700 --task-end 700
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
