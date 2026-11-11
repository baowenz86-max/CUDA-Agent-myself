# CUDA Agent Workflow Report

- Task ID: `750`
- Status: **FAILED**
- Operators: `["F.softmax", "F.cross_entropy"]`
- Started: `2026-11-11T13:26:35`
- Finished: `2026-11-11T13:31:59`
- Duration: `324.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.32s |  |
| generation #1 | PASS | 52.62s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; custom_softmax: py::arg order/name/default differs |
| repair API call #1 | PASS | 14.41s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 19.30s | best; pipeline_progress; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0750/kernels/custom_ops_binding.cpp:108:33: error: invalid conversion from ‘long int*’ to ‘const long long int*’ [-fpermissive]
  108 \|         target.data_ptr<int64_t>(),  // int64_t is long long
      \|         ~~~~~~~~~~~~~~~~~~~~~~~~^~
      \|                                 \| |
| repair API call #2 | PASS | 26.66s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 89.01s | best; pipeline_progress; AssertionError: Scalars are not close!

Expected 2.30755352973938 but got 22882824.0.
Absolute difference: 22882821.69244647 (up to 0.01 allowed)
Expected 2.307662010192871 but got 22891046.0.
Absolute difference: 22891043.69233799 (up to 0.01 allowed)
Expected 2.307673454284668 but got 22888412.0.
Absolute difference: 22888409.692326546 (up to 0.01 allowed)
Expected 2.3076887130737305 but got 22892130.0.
Absolute difference: 22892127.692311287 (up to 0.01 allowed)
Expected 2.3076109886169434 but got 22888066.0.
Absolute difference: 22888063.69238901 (up to 0.01 allowed) |
| repair API call #3 | PASS | 35.25s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 82.51s | best; final_progress; AssertionError: Scalars are not close!

Expected 2.307553768157959 but got 2.72784423828125.
Absolute difference: 0.420290470123291 (up to 0.01 allowed)
Expected 2.307661771774292 but got 2.7288260459899902.
Absolute difference: 0.42116427421569824 (up to 0.01 allowed)
Expected 2.307673215866089 but got 2.7285118103027344.
Absolute difference: 0.4208385944366455 (up to 0.01 allowed)
Expected 2.307687759399414 but got 2.728956937789917.
Absolute difference: 0.42126917839050293 (up to 0.01 allowed)
Expected 2.3076109886169434 but got 2.7284703254699707.
Absolute difference: 0.42085933685302734 (up to 0.01 allowed) |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [], "dtype": "torch.float32", "device": "cuda:0", "stride": [], "numel": 1}, "expected": {"type": "tensor", "shape": [], "dtype": "torch.float32", "device": "cuda:0", "stride": [], "numel": 1}, "kind": "tensor_values", "mismatched": 1, "numel": 1, "actual_nan": 0, "expected_nan": 0, "actual_inf": 0, "expected_inf": 0, "mismatch_ratio": 1.0, "max_absolute_error": 0.420290470123291, "max_relative_error": 0.18213680475094388, "samples": [{"index": [], "actual": "2.72784423828125", "expected": "2.307553768157959"}], "path": "output", "message": "Scalars are not close!\n\nExpected 2.307553768157959 but got 2.72784423828125.\nAbsolute difference: 0.420290470123291 (up to 0.01 allowed)\nRelative difference: 0.18213680475094388 (up to 0.01 allowed)"}

## Reproduction

```bash
python main.py --task-start 750 --task-end 750
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
