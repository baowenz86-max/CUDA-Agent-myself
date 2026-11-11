# CUDA Agent Workflow Report

- Task ID: `515`
- Status: **SUCCESS**
- Operators: `["nn.AdaptiveAvgPool3d"]`
- Started: `2026-11-10T18:02:56`
- Finished: `2026-11-10T18:11:39`
- Duration: `523.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.38s |  |
| generation #1 | PASS | 80.28s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 92.69s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 64 / 64 (100.0%)
Greatest absolute difference: 0.2721205949783325 at index (0, 14) (up to 0.01 allowed)
Mismatched elements: 31 / 64 (48.4%)
Greatest absolute difference: 3.6893488147419103e+19 at index (2, 6) (up to 0.01 allowed)
Mismatched elements: 29 / 64 (45.3%)
Greatest absolute difference: 4.253529586511731e+37 at index (2, 12) (up to 0.01 allowed)
Mismatched elements: 26 / 64 (40.6%)
Greatest absolute difference: 1.5845632502852868e+29 at index (3, 10) (up to 0.01 allowed) |
| repair API call #1 | PASS | 158.35s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 16.23s | rejected; tie_keep_previous; AssertionError: Tensor-likes are not close!

Mismatched elements: 64 / 64 (100.0%)
Greatest absolute difference: 0.2721205949783325 at index (0, 14) (up to 0.01 allowed)
Mismatched elements: 31 / 64 (48.4%)
Greatest absolute difference: 3.6893488147419103e+19 at index (2, 6) (up to 0.01 allowed)
Mismatched elements: 29 / 64 (45.3%)
Greatest absolute difference: 4.253529586511731e+37 at index (2, 12) (up to 0.01 allowed)
Mismatched elements: 26 / 64 (40.6%)
Greatest absolute difference: 1.5845632502852868e+29 at index (3, 10) (up to 0.01 allowed) |
| repair API call #2 | PASS | 127.49s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 12.12s | best; correctness_success |
| profiling | PASS | 10.48s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 515 --task-end 515
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
