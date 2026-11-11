# CUDA Agent Workflow Report

- Task ID: `460`
- Status: **FAILED**
- Operators: `["torch.unbind", "nn.PixelShuffle"]`
- Started: `2026-11-10T15:09:42`
- Finished: `2026-11-10T15:30:30`
- Duration: `1247.61s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 79.05s |  |
| generation #1 | PASS | 281.43s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 24.22s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0460/kernels/custom_ops.cu(145): error: function "stack_dim1_launcher" has already been defined (previous definition at line 76)
  extern "C" void stack_dim1_launcher(const float* const* input_ptrs,
                  ^
 |
| repair API call #1 | PASS | 20.88s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 30.73s | best; pipeline_progress; RuntimeError: x must be contiguous

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0460/20261110_150942_785053/correctness/verification.json |
| repair API call #2 | PASS | 34.36s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 39.96s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 266032326 / 268435456 (99.1%)
Greatest absolute difference: 8.726953506469727 at index (338, 24, 91, 94) (up to 0.01 allowed)
Mismatched elements: 266033342 / 268435456 (99.1%)
Greatest absolute difference: 8.398479461669922 at index (84, 10, 86, 120) (up to 0.01 allowed)
Mismatched elements: 266034567 / 268435456 (99.1%)
Greatest absolute difference: 8.322324752807617 at index (401, 9, 31, 29) (up to 0.01 allowed)
Mismatched elements: 266035218 / 268435456 (99.1%)
Greatest absolute difference: 8.111227989196777 at index (508, 16, 14, 9) (up to 0.01 allowed)
Mismatched elements: 266031087 / 268435456 (99.1%)
Greatest absolute difference: 8.974715232849121 at index (211, 16, 76, 125) (up to 0.01 allowed) |
| repair API call #3 | FAIL | 587.86s | Code repairs used: 2/3; base candidate #3; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #4 | PASS | 105.41s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 40.22s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 266032326 / 268435456 (99.1%)
Greatest absolute difference: 8.726953506469727 at index (338, 24, 91, 94) (up to 0.01 allowed)
Mismatched elements: 266033342 / 268435456 (99.1%)
Greatest absolute difference: 8.398479461669922 at index (84, 10, 86, 120) (up to 0.01 allowed)
Mismatched elements: 266034567 / 268435456 (99.1%)
Greatest absolute difference: 8.322324752807617 at index (401, 9, 31, 29) (up to 0.01 allowed)
Mismatched elements: 266035218 / 268435456 (99.1%)
Greatest absolute difference: 8.111227989196777 at index (508, 16, 14, 9) (up to 0.01 allowed)
Mismatched elements: 266031087 / 268435456 (99.1%)
Greatest absolute difference: 8.974715232849121 at index (211, 16, 76, 125) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [512, 32, 128, 128], "dtype": "torch.float32", "device": "cuda:0", "stride": [524288, 16384, 128, 1], "numel": 268435456}, "expected": {"type": "tensor", "shape": [512, 32, 128, 128], "dtype": "torch.float32", "device": "cuda:0", "stride": [524288, 16384, 128, 1], "numel": 268435456}, "kind": "tensor_values", "mismatched": 266032326, "numel": 268435456, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9910476431250572, "max_absolute_error": 8.726953983306885, "max_relative_error": 2.779997825622558e+30, "samples": [{"index": [0, 1, 0, 0], "actual": "-1.28819739818573", "expected": "-0.09227024018764496"}, {"index": [0, 1, 0, 1], "actual": "-0.4992426037788391", "expected": "-0.11670452356338501"}, {"index": [0, 1, 0, 2], "actual": "0.571447491645813", "expected": "1.397882342338562"}, {"index": [0, 1, 0, 3], "actual": "-1.1379247903823853", "expected": "-0.9611736536026001"}, {"index": [0, 1, 0, 4], "actual": "-1.140820860862732", "expected": "2.0928943157196045"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 266032326 / 268435456 (99.1%)\nGreatest absolute difference: 8.726953506469727 at index (338, 24, 91, 94) (up to 0.01 allowed)\nGreatest relative difference: inf at index (15, 7, 18, 2) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 460 --task-end 460
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
