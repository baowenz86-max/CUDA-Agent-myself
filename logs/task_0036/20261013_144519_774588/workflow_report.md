# CUDA Agent Workflow Report

- Task ID: `36`
- Status: **FAILED**
- Operators: `["torch.add", "torch.argmin", "torch.stft"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T15:03:49`
- Duration: `1109.30s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0005`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.96s |  |
| generation #1 | PASS | 93.34s | DeepSeek API |
| compile candidate #1 | FAIL | 53.76s | best |
| compile candidate #2 | FAIL | 50.27s | rejected |
| verification candidate #3 | FAIL | 35.01s | best |
| generation #2 | PASS | 140.84s | DeepSeek API |
| compile candidate #4 | FAIL | 32.87s | rejected |
| verification candidate #5 | FAIL | 53.38s | best |
| verification candidate #6 | FAIL | 59.42s | rejected |
| generation #3 | PASS | 141.85s | DeepSeek API |
| compile candidate #7 | FAIL | 12.18s | rejected |
| verification candidate #8 | FAIL | 47.47s | rejected |
| verification candidate #9 | FAIL | 29.17s | rejected |

## Failure Summary

Best candidate 5, stage verification: {"actual": {"type": "tensor", "shape": [128, 513, 782], "dtype": "torch.complex64", "device": "cuda:0", "stride": [401166, 782, 1], "numel": 51349248}, "expected": {"type": "tensor", "shape": [128, 513, 782], "dtype": "torch.complex64", "device": "cuda:0", "stride": [401166, 1, 513], "numel": 51349248}, "kind": "tensor_values", "mismatched": 51343306, "numel": 51349248, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9998842826286375, "max_absolute_error": 188.14812139568397, "max_relative_error": 196213.22931447526, "samples": [{"index": [0, 0, 0], "actual": "(-43.60857391357422+0j)", "expected": "(3.3297128677368164+0j)"}, {"index": [0, 0, 1], "actual": "(-6.265471935272217+0j)", "expected": "(-33.462669372558594+0j)"}, {"index": [0, 0, 2], "actual": "(1.9390878677368164+0j)", "expected": "(-25.175003051757812+0j)"}, {"index": [0, 0, 3], "actual": "(-22.9298095703125+0j)", "expected": "(-25.49956512451172+0j)"}, {"index": [0, 0, 4], "actual": "(-15.44317626953125+0j)", "expected": "(-11.782236099243164+0j)"}], "path": "output[2]", "message": "Tensor-likes are not close!\n\nMismatched elements: 51343306 / 51349248 (100.0%)\nGreatest absolute difference: 188.1481170654297 at index (74, 507, 781) (up to 0.01 allowed)\nGreatest relative difference: 196213.25 at index (28, 34, 0) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 36 --task-end 36
```
