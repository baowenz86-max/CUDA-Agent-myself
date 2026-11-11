# CUDA Agent Workflow Report

- Task ID: `771`
- Status: **SUCCESS**
- Operators: `["torch.isin", "nn.LSTM"]`
- Started: `2026-11-11T14:05:37`
- Finished: `2026-11-11T14:09:39`
- Duration: `241.94s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 33.77s |  |
| generation #1 | PASS | 117.50s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 32.10s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 852285 / 1048576 (81.3%)
Greatest absolute difference: 0.15957584977149963 at index (14, 840, 28) (up to 0.01 allowed)
Mismatched elements: 851004 / 1048576 (81.2%)
Greatest absolute difference: 0.15591871738433838 at index (9, 99, 28) (up to 0.01 allowed)
Mismatched elements: 851930 / 1048576 (81.2%)
Greatest absolute difference: 0.15448333323001862 at index (24, 442, 28) (up to 0.01 allowed)
Mismatched elements: 852685 / 1048576 (81.3%)
Greatest absolute difference: 0.1704043447971344 at index (1, 561, 28) (up to 0.01 allowed)
Mismatched elements: 851991 / 1048576 (81.3%)
Greatest absolute difference: 0.16575710475444794 at index (30, 15, 27) (up to 0.01 allowed) |
| repair API call #1 | PASS | 38.45s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 13.12s | best; correctness_success |
| profiling | PASS | 5.40s |  |
| artifacts | PASS | 1.46s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 771 --task-end 771
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
