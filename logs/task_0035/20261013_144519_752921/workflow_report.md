# CUDA Agent Workflow Report

- Task ID: `35`
- Status: **FAILED**
- Operators: `["nn.ParameterDict", "nn.Upsample", "torch.logical_and"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T15:00:16`
- Duration: `896.71s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.02s |  |
| generation #1 | PASS | 73.86s | DeepSeek API |
| verification candidate #1 | FAIL | 41.43s | best |
| verification candidate #2 | FAIL | 64.18s | rejected |
| verification candidate #3 | FAIL | 52.44s | rejected |
| generation #2 | PASS | 65.81s | DeepSeek API |
| verification candidate #4 | FAIL | 75.10s | rejected |
| verification candidate #5 | FAIL | 34.24s | rejected |
| verification candidate #6 | FAIL | 33.45s | rejected |
| generation #3 | PASS | 86.44s | DeepSeek API |
| verification candidate #7 | FAIL | 76.97s | rejected |
| verification candidate #8 | FAIL | 27.86s | rejected |
| verification candidate #9 | FAIL | 42.54s | rejected |

## Failure Summary

Best candidate 1, stage verification: {"actual": {"type": "tensor", "shape": [16, 3, 1024, 2048], "dtype": "torch.bool", "device": "cuda:0", "stride": [6291456, 2097152, 2048, 1], "numel": 100663296}, "expected": {"type": "tensor", "shape": [16, 3, 1024, 2048], "dtype": "torch.bool", "device": "cuda:0", "stride": [6291456, 2097152, 2048, 1], "numel": 100663296}, "kind": "tensor_values", "mismatched": 1, "numel": 100663296, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 9.934107462565104e-09, "max_absolute_error": 1.0, "max_relative_error": 1.0, "samples": [{"index": [2, 2, 254, 601], "actual": "False", "expected": "True"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 1 / 100663296 (0.0%)\nGreatest absolute difference: 1 at index (2, 2, 254, 601) (up to 0.01 allowed)\nGreatest relative difference: 1.0 at index (2, 2, 254, 601) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 35 --task-end 35
```
