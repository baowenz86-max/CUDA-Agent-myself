# CUDA Agent Workflow Report

- Task ID: `20`
- Status: **FAILED**
- Operators: `["nn.ConvTranspose1d", "torch.sign", "nn.Embedding"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:45:10`
- Duration: `535.12s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.68s |  |
| generation #1 | PASS | 45.59s | DeepSeek API |
| verification candidate #1 | FAIL | 41.81s | best |
| repair API | FAIL | 0.00s | No changes applied; best candidate retained |
| generation #2 | FAIL | 42.98s | DeepSeek API |
| generation #3 | PASS | 53.55s | DeepSeek API |
| compile candidate #3 | FAIL | 137.83s | rejected |
| verification candidate #4 | FAIL | 63.96s | rejected |
| verification candidate #5 | FAIL | 24.06s | rejected |

## Failure Summary

Best candidate 1, stage verification: {"actual": {"type": "tensor", "shape": [2048, 16, 512], "dtype": "torch.float32", "device": "cuda:0", "stride": [8192, 512, 1], "numel": 16777216}, "expected": {"type": "tensor", "shape": [2048, 16, 512], "dtype": "torch.float32", "device": "cuda:0", "stride": [8192, 512, 1], "numel": 16777216}, "kind": "tensor_values", "mismatched": 1029, "numel": 16777216, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 6.133317947387695e-05, "max_absolute_error": 2.0, "max_relative_error": 2.0, "samples": [{"index": [0, 9, 121], "actual": "-1.0", "expected": "1.0"}, {"index": [4, 11, 193], "actual": "1.0", "expected": "-1.0"}, {"index": [11, 0, 451], "actual": "-1.0", "expected": "1.0"}, {"index": [12, 10, 63], "actual": "1.0", "expected": "-1.0"}, {"index": [13, 9, 190], "actual": "-1.0", "expected": "1.0"}], "path": "output[0]", "message": "Tensor-likes are not close!\n\nMismatched elements: 1029 / 16777216 (0.0%)\nGreatest absolute difference: 2.0 at index (0, 9, 121) (up to 0.01 allowed)\nGreatest relative difference: 2.0 at index (0, 9, 121) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 20 --task-end 20
```
