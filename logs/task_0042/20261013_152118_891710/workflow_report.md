# CUDA Agent Workflow Report

- Task ID: `42`
- Status: **FAILED**
- Operators: `["nn.MultiheadAttention", "torch.cat", "F.smooth_l1_loss"]`
- Started: `2026-10-13T15:21:18`
- Finished: `2026-10-13T15:42:44`
- Duration: `1285.94s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.09s |  |
| generation #1 | PASS | 88.96s | DeepSeek API |
| compile candidate #1 | FAIL | 20.54s | best |
| verification candidate #2 | FAIL | 20.32s | best |
| compile candidate #3 | FAIL | 2.92s | rejected |
| generation #2 | PASS | 251.20s | DeepSeek API |
| compile candidate #4 | FAIL | 12.28s | rejected |
| compile candidate #5 | FAIL | 16.85s | rejected |
| compile candidate #6 | FAIL | 2.80s | rejected |
| generation #3 | PASS | 285.81s | DeepSeek API |
| compile candidate #7 | FAIL | 16.35s | rejected |
| compile candidate #8 | FAIL | 19.45s | rejected |
| compile candidate #9 | FAIL | 39.88s | rejected |

## Failure Summary

Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [16, 1024, 512], "dtype": "torch.float32", "device": "cuda:0", "stride": [524288, 512, 1], "numel": 8388608}, "expected": {"type": "tensor", "shape": [16, 1024, 512], "dtype": "torch.float32", "device": "cuda:0", "stride": [524288, 512, 1], "numel": 8388608}, "kind": "tensor_values", "mismatched": 8277774, "numel": 8388608, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 0.9867875576019287, "max_absolute_error": 5.582678318023682, "max_relative_error": 101524161.64647591, "samples": [{"index": [0, 0, 0], "actual": "-0.414092093706131", "expected": "0.021615097299218178"}, {"index": [0, 0, 1], "actual": "-0.4098528325557709", "expected": "0.0007492195582017303"}, {"index": [0, 0, 2], "actual": "-0.6494219899177551", "expected": "-0.012401806190609932"}, {"index": [0, 0, 3], "actual": "-0.370454877614975", "expected": "-0.026422088965773582"}, {"index": [0, 0, 4], "actual": "0.09217632561922073", "expected": "0.0015558652812615037"}], "path": "output[0]", "message": "Tensor-likes are not close!\n\nMismatched elements: 8277774 / 8388608 (98.7%)\nGreatest absolute difference: 5.582678318023682 at index (14, 739, 390) (up to 0.01 allowed)\nGreatest relative difference: 101524160.0 at index (9, 456, 231) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 42 --task-end 42
```
