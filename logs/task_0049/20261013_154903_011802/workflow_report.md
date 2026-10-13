# CUDA Agent Workflow Report

- Task ID: `49`
- Status: **FAILED**
- Operators: `["F.conv_transpose1d", "nn.ParameterList", "torch.round"]`
- Started: `2026-10-13T15:49:03`
- Finished: `2026-10-13T16:07:58`
- Duration: `1135.95s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 33.94s |  |
| generation #1 | PASS | 156.64s | DeepSeek API |
| compile candidate #1 | FAIL | 9.57s | best |
| verification candidate #2 | FAIL | 35.21s | best |
| verification candidate #3 | FAIL | 15.49s | best |
| generation #2 | PASS | 96.03s | DeepSeek API |
| compile candidate #4 | FAIL | 23.21s | rejected |
| compile candidate #5 | FAIL | 4.78s | rejected |
| compile candidate #6 | FAIL | 4.76s | rejected |
| generation #3 | PASS | 63.48s | DeepSeek API |
| compile candidate #7 | FAIL | 4.84s | rejected |
| compile candidate #8 | FAIL | 4.84s | rejected |
| compile candidate #9 | FAIL | 4.79s | rejected |

## Failure Summary

Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [128, 8, 16384], "dtype": "torch.float32", "device": "cuda:0", "stride": [131072, 16384, 1], "numel": 16777216}, "expected": {"type": "tensor", "shape": [128, 8, 16384], "dtype": "torch.float32", "device": "cuda:0", "stride": [131072, 16384, 1], "numel": 16777216}, "kind": "tensor_values", "mismatched": 13343925, "numel": 16777216, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.7953599095344543, "max_absolute_error": 9.0, "max_relative_error": 7.999999999999999e+30, "samples": [{"index": [0, 0, 2], "actual": "-1.0", "expected": "-0.0"}, {"index": [0, 0, 3], "actual": "-0.0", "expected": "-1.0"}, {"index": [0, 0, 4], "actual": "-1.0", "expected": "-0.0"}, {"index": [0, 0, 5], "actual": "-0.0", "expected": "-1.0"}, {"index": [0, 0, 6], "actual": "-1.0", "expected": "-0.0"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 13343925 / 16777216 (79.5%)\nGreatest absolute difference: 9.0 at index (126, 6, 2047) (up to 0.01 allowed)\nGreatest relative difference: inf at index (0, 0, 2) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 49 --task-end 49
```
