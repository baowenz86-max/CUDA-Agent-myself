# CUDA Agent Workflow Report

- Task ID: `331`
- Status: **FAILED**
- Operators: `["F.gelu", "torch.gt"]`
- Started: `2026-11-07T16:56:26`
- Finished: `2026-11-07T17:00:07`
- Duration: `220.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.52s |  |
| generation #1 | PASS | 77.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 30.84s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 65138688 / 301989888 (21.6%)
Greatest absolute difference: 1.0000001192092896 at index (54, 0, 1) (up to 0.01 allowed)
Mismatched elements: 65089536 / 301989888 (21.6%)
Greatest absolute difference: 1.0000001192092896 at index (54, 0, 3) (up to 0.01 allowed)
Mismatched elements: 65498112 / 301989888 (21.7%)
Greatest absolute difference: 1.0000001192092896 at index (54, 0, 2) (up to 0.01 allowed)
Mismatched elements: 65074176 / 301989888 (21.5%)
Mismatched elements: 65384448 / 301989888 (21.7%) |
| repair API call #1 | PASS | 14.27s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 27.58s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 1536 / 301989888 (0.0%)
Greatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed) |
| repair API call #2 | PASS | 22.95s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 13.68s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 1536 / 301989888 (0.0%)
Greatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed) |
| repair API call #3 | PASS | 16.24s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 13.83s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 1536 / 301989888 (0.0%)
Greatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [1536, 128, 1536], "dtype": "torch.float32", "device": "cuda:0", "stride": [196608, 1536, 1], "numel": 301989888}, "expected": {"type": "tensor", "shape": [1536, 128, 1536], "dtype": "torch.float32", "device": "cuda:0", "stride": [196608, 1536, 1], "numel": 301989888}, "kind": "tensor_values", "mismatched": 1536, "numel": 301989888, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 5.086263020833333e-06, "max_absolute_error": 1.0000001192092896, "max_relative_error": 1854.6599618075322, "samples": [{"index": [0, 58, 1038], "actual": "0.7070679068565369", "expected": "-0.29293209314346313"}, {"index": [1, 58, 1038], "actual": "1.025524616241455", "expected": "0.0255245603621006"}, {"index": [2, 58, 1038], "actual": "1.608433723449707", "expected": "0.6084337830543518"}, {"index": [3, 58, 1038], "actual": "0.9328385591506958", "expected": "-0.06716146320104599"}, {"index": [4, 58, 1038], "actual": "0.6356709599494934", "expected": "-0.3643290400505066"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 1536 / 301989888 (0.0%)\nGreatest absolute difference: 1.0000001192092896 at index (54, 58, 1038) (up to 0.01 allowed)\nGreatest relative difference: 1854.659912109375 at index (1274, 58, 1038) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 331 --task-end 331
```
