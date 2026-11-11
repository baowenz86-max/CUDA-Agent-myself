# CUDA Agent Workflow Report

- Task ID: `363`
- Status: **SUCCESS**
- Operators: `["F.pixel_unshuffle"]`
- Started: `2026-11-09T14:59:56`
- Finished: `2026-11-09T15:06:33`
- Duration: `397.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.36s |  |
| generation #1 | PASS | 43.34s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 128.27s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 332583215 / 402653184 (82.6%)
Greatest absolute difference: 8.165700912475586 at index (385, 7, 181, 42) (up to 0.01 allowed)
Mismatched elements: 332581290 / 402653184 (82.6%)
Greatest absolute difference: 8.165093421936035 at index (487, 5, 80, 126) (up to 0.01 allowed)
Mismatched elements: 332580588 / 402653184 (82.6%)
Greatest absolute difference: 8.334758758544922 at index (202, 10, 197, 106) (up to 0.01 allowed)
Mismatched elements: 332583242 / 402653184 (82.6%)
Greatest absolute difference: 8.347917556762695 at index (30, 7, 62, 68) (up to 0.01 allowed)
Mismatched elements: 332586224 / 402653184 (82.6%)
Greatest absolute difference: 8.53475284576416 at index (299, 8, 75, 42) (up to 0.01 allowed) |
| repair API call #1 | PASS | 21.39s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 158.61s | best; correctness_success |
| profiling | PASS | 36.49s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 363 --task-end 363
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
