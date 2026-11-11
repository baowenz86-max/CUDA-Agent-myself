# CUDA Agent Workflow Report

- Task ID: `839`
- Status: **FAILED**
- Operators: `["torch.log_softmax", "torch.tanh", "torch.bmm"]`
- Started: `2026-11-11T17:58:22`
- Finished: `2026-11-11T18:10:43`
- Duration: `740.90s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 52.95s |  |
| generation #1 | PASS | 182.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 36.03s | best; initial_candidate; RuntimeError: h0 batch dimension must match x batch size

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0839/20261111_175822_410158/correctness/verification.json |
| repair API call #1 | PASS | 152.10s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 13.49s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 5 / 5 (100.0%)
Greatest absolute difference: 0.07916560769081116 at index (2, 0) (up to 0.01 allowed)
Greatest absolute difference: 0.06764719635248184 at index (2, 0) (up to 0.01 allowed)
Greatest absolute difference: 1.4438731670379639 at index (3, 0) (up to 0.01 allowed)
Greatest absolute difference: 0.09081343561410904 at index (1, 0) (up to 0.01 allowed)
Greatest absolute difference: 1.5020575523376465 at index (3, 0) (up to 0.01 allowed) |
| repair API call #2 | PASS | 192.99s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 14.89s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 5 / 5 (100.0%)
Greatest absolute difference: 0.07916560769081116 at index (2, 0) (up to 0.01 allowed)
Greatest absolute difference: 0.06764719635248184 at index (2, 0) (up to 0.01 allowed)
Greatest absolute difference: 1.4438731670379639 at index (3, 0) (up to 0.01 allowed)
Greatest absolute difference: 0.09081343561410904 at index (1, 0) (up to 0.01 allowed)
Greatest absolute difference: 1.5020575523376465 at index (3, 0) (up to 0.01 allowed) |
| repair API call #3 | PASS | 74.94s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 20.89s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 5 / 5 (100.0%)
Greatest absolute difference: 0.07916560769081116 at index (2, 0) (up to 0.01 allowed)
Greatest absolute difference: 0.06764719635248184 at index (2, 0) (up to 0.01 allowed)
Greatest absolute difference: 1.4438731670379639 at index (3, 0) (up to 0.01 allowed)
Greatest absolute difference: 0.09081343561410904 at index (1, 0) (up to 0.01 allowed)
Greatest absolute difference: 1.5020575523376465 at index (3, 0) (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"actual": {"type": "tensor", "shape": [5, 1], "dtype": "torch.float32", "device": "cuda:0", "stride": [1, 1], "numel": 5}, "expected": {"type": "tensor", "shape": [5, 1], "dtype": "torch.float32", "device": "cuda:0", "stride": [1, 1], "numel": 5}, "kind": "tensor_values", "mismatched": 5, "numel": 5, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 1.0, "max_absolute_error": 0.07916560769081116, "max_relative_error": 1.0, "samples": [{"index": [0, 0], "actual": "-0.052254222333431244", "expected": "-0.0734185203909874"}, {"index": [1, 0], "actual": "0.0", "expected": "-0.06016423925757408"}, {"index": [2, 0], "actual": "0.0", "expected": "-0.07916560769081116"}, {"index": [3, 0], "actual": "0.0", "expected": "-0.061078302562236786"}, {"index": [4, 0], "actual": "0.0", "expected": "-0.06992217153310776"}], "path": "output", "message": "Tensor-likes are not close!\n\nMismatched elements: 5 / 5 (100.0%)\nGreatest absolute difference: 0.07916560769081116 at index (2, 0) (up to 0.01 allowed)\nGreatest relative difference: 1.0 at index (1, 0) (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 839 --task-end 839
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
