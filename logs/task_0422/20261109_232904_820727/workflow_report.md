# CUDA Agent Workflow Report

- Task ID: `422`
- Status: **SUCCESS**
- Operators: `["torch.max", "nn.MSELoss"]`
- Started: `2026-11-09T23:29:04`
- Finished: `2026-11-09T23:34:34`
- Duration: `329.30s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.02s |  |
| generation #1 | PASS | 54.90s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 196.24s | best; initial_candidate; AssertionError: Scalars are not close!

Expected 4.554629325866699 but got 152827968.0.
Absolute difference: 152827963.44537067 (up to 0.01 allowed)
Expected 4.558769226074219 but got 152966880.0.
Absolute difference: 152966875.44123077 (up to 0.01 allowed)
Expected 4.312171936035156 but got 144692464.0.
Absolute difference: 144692459.68782806 (up to 0.01 allowed)
Expected 4.507096767425537 but got 151233072.0.
Absolute difference: 151233067.49290323 (up to 0.01 allowed)
Expected 4.429405212402344 but got 148626192.0.
Absolute difference: 148626187.5705948 (up to 0.01 allowed) |
| repair API call #1 | PASS | 15.33s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 32.10s | best; correctness_success |
| profiling | PASS | 14.90s |  |
| artifacts | PASS | 1.16s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 422 --task-end 422
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
