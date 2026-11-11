# CUDA Agent Workflow Report

- Task ID: `828`
- Status: **SUCCESS**
- Operators: `["torch.reshape", "torch.stft"]`
- Started: `2026-11-11T17:49:13`
- Finished: `2026-11-11T17:54:46`
- Duration: `332.61s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 39.50s |  |
| generation #1 | PASS | 125.26s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.05s | best; initial_candidate; custom_reshape_output.shape: C++ type/nullable differs |
| repair API call #1 | PASS | 34.98s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 44.16s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 69204012 / 69239808 (99.9%)
Greatest absolute difference: 465.10321044921875 at index (880, 34, 0) (up to 0.01 allowed)
Mismatched elements: 69203987 / 69239808 (99.9%)
Greatest absolute difference: 433.1498107910156 at index (957, 1410, 1) (up to 0.01 allowed)
Mismatched elements: 69203894 / 69239808 (99.9%)
Greatest absolute difference: 452.8399353027344 at index (979, 2048, 0) (up to 0.01 allowed)
Mismatched elements: 69203886 / 69239808 (99.9%)
Greatest absolute difference: 434.6368713378906 at index (762, 1677, 32) (up to 0.01 allowed)
Mismatched elements: 69203994 / 69239808 (99.9%)
Greatest absolute difference: 430.0876770019531 at index (497, 2048, 29) (up to 0.01 allowed) |
| repair API call #2 | PASS | 32.58s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 31.57s | best; correctness_success |
| profiling | PASS | 17.09s |  |
| artifacts | PASS | 7.13s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 828 --task-end 828
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
