# CUDA Agent Workflow Report

- Task ID: `411`
- Status: **FAILED**
- Operators: `["nn.Embedding", "F.kl_div"]`
- Started: `2026-11-09T22:58:51`
- Finished: `2026-11-09T23:04:49`
- Duration: `358.35s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.45s |  |
| generation #1 | PASS | 109.44s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 40.65s | best; initial_candidate; RuntimeError: input must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0411/20261109_225851_071354/correctness/verification.json |
| repair API call #1 | PASS | 57.86s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 32.00s | best; pipeline_progress; AssertionError: Scalars are not close!

Expected 1018.8927612304688 but got 24033.265625.
Absolute difference: 23014.37286376953 (up to 0.01 allowed)
Expected 1019.0015258789062 but got 24032.5546875.
Absolute difference: 23013.553161621094 (up to 0.01 allowed)
Expected 1019.0543212890625 but got 24038.513671875.
Absolute difference: 23019.459350585938 (up to 0.01 allowed)
Expected 1019.0142211914062 but got 24039.400390625.
Absolute difference: 23020.386169433594 (up to 0.01 allowed)
Expected 1019.3690185546875 but got 24037.30078125.
Absolute difference: 23017.931762695312 (up to 0.01 allowed) |
| repair API call #2 | PASS | 38.65s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 18.98s | rejected; final_mismatch_regression; AssertionError: Scalars are not close!

Expected 1018.8927612304688 but got nan.
Absolute difference: nan (up to 0.01 allowed)
Expected 1019.0015258789062 but got nan.
Expected 1019.0543212890625 but got nan.
Expected 1019.0142211914062 but got nan.
Expected 1019.3690185546875 but got nan. |
| repair API call #3 | PASS | 34.95s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 19.22s | best; final_progress; AssertionError: Scalars are not close!

Expected 1018.8927612304688 but got 24033.14453125.
Absolute difference: 23014.25177001953 (up to 0.01 allowed)
Expected 1019.0015258789062 but got 24032.4296875.
Absolute difference: 23013.428161621094 (up to 0.01 allowed)
Expected 1019.0543212890625 but got 24038.50390625.
Absolute difference: 23019.449584960938 (up to 0.01 allowed)
Expected 1019.0142211914062 but got 24039.318359375.
Absolute difference: 23020.304138183594 (up to 0.01 allowed)
Expected 1019.3690185546875 but got 24037.0390625.
Absolute difference: 23017.670043945312 (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [], "dtype": "torch.float32", "device": "cuda:0", "stride": [], "numel": 1}, "expected": {"type": "tensor", "shape": [], "dtype": "torch.float32", "device": "cuda:0", "stride": [], "numel": 1}, "kind": "tensor_values", "mismatched": 1, "numel": 1, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 1.0, "max_absolute_error": 23014.25177001953, "max_relative_error": 22.58751131201119, "samples": [{"index": [], "actual": "24033.14453125", "expected": "1018.8927612304688"}], "path": "output", "message": "Scalars are not close!\n\nExpected 1018.8927612304688 but got 24033.14453125.\nAbsolute difference: 23014.25177001953 (up to 0.01 allowed)\nRelative difference: 22.58751131201119 (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 411 --task-end 411
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
