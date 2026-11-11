# CUDA Agent Workflow Report

- Task ID: `837`
- Status: **FAILED**
- Operators: `["nn.AdaptiveLogSoftmaxWithLoss", "torch.unsqueeze"]`
- Started: `2026-11-11T17:58:22`
- Finished: `2026-11-11T18:12:13`
- Duration: `831.32s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.08s |  |
| generation #1 | PASS | 94.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 78.48s | best; initial_candidate; AssertionError: The values for attribute 'shape' do not match: torch.Size([2048, 1, 10001]) != torch.Size([2048, 1, 10000]).

The above exception was the direct cause of the following exception: |
| repair API call #1 | PASS | 61.06s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 42.93s | best; final_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 10240000 / 20480000 (50.0%)
Greatest absolute difference: 10.748784065246582 at index (332, 0, 5022) (up to 0.01 allowed)
Greatest absolute difference: 10.449026107788086 at index (439, 0, 5303) (up to 0.01 allowed)
Greatest absolute difference: 10.707862854003906 at index (741, 0, 5022) (up to 0.01 allowed)
Greatest absolute difference: 10.782475471496582 at index (375, 0, 5008) (up to 0.01 allowed)
Greatest absolute difference: 10.523734092712402 at index (1487, 0, 5007) (up to 0.01 allowed) |
| repair API call #2 | FAIL | 197.95s | Code repairs used: 1/3; base candidate #2; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #3 | PASS | 225.55s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 33.57s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 10240000 / 20480000 (50.0%)
Greatest absolute difference: 10.748784065246582 at index (332, 0, 5022) (up to 0.01 allowed)
Greatest absolute difference: 10.449026107788086 at index (439, 0, 5303) (up to 0.01 allowed)
Greatest absolute difference: 10.707862854003906 at index (741, 0, 5022) (up to 0.01 allowed)
Greatest absolute difference: 10.782475471496582 at index (375, 0, 5008) (up to 0.01 allowed)
Greatest absolute difference: 10.523734092712402 at index (1487, 0, 5007) (up to 0.01 allowed) |
| repair API call #4 | PASS | 46.64s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 27.61s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 10240000 / 20480000 (50.0%)
Greatest absolute difference: 10.748784065246582 at index (332, 0, 5022) (up to 0.01 allowed)
Greatest absolute difference: 10.449026107788086 at index (439, 0, 5303) (up to 0.01 allowed)
Greatest absolute difference: 10.707862854003906 at index (741, 0, 5022) (up to 0.01 allowed)
Greatest absolute difference: 10.782475471496582 at index (375, 0, 5008) (up to 0.01 allowed)
Greatest absolute difference: 10.523734092712402 at index (1487, 0, 5007) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [2048, 1, 10000], "dtype": "torch.float32", "device": "cuda:0", "stride": [10000, 10000, 1], "numel": 20480000}, "expected": {"type": "tensor", "shape": [2048, 1, 10000], "dtype": "torch.float32", "device": "cuda:0", "stride": [10000, 10000, 1], "numel": 20480000}, "kind": "tensor_values", "mismatched": 10240000, "numel": 20480000, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.5, "max_absolute_error": 10.748784065246582, "max_relative_error": 0.59991608243913, "samples": [{"index": [0, 0, 5000], "actual": "-8.466080665588379", "expected": "-16.124618530273438"}, {"index": [0, 0, 5001], "actual": "-8.67687702178955", "expected": "-16.33541488647461"}, {"index": [0, 0, 5002], "actual": "-8.374288558959961", "expected": "-16.032825469970703"}, {"index": [0, 0, 5003], "actual": "-8.305864334106445", "expected": "-15.964402198791504"}, {"index": [0, 0, 5004], "actual": "-8.55099105834961", "expected": "-16.209529876708984"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 10240000 / 20480000 (50.0%)\nGreatest absolute difference: 10.748784065246582 at index (332, 0, 5022) (up to 0.01 allowed)\nGreatest relative difference: 0.5999161005020142 at index (332, 0, 8993) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 837 --task-end 837
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
