# CUDA Agent Workflow Report

- Task ID: `314`
- Status: **SUCCESS**
- Operators: `["F.leaky_relu", "nn.AvgPool1d"]`
- Started: `2026-11-07T15:46:50`
- Finished: `2026-11-07T16:04:47`
- Duration: `1076.51s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.78s |  |
| generation #1 | PASS | 610.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 101.47s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 70950 / 28311552 (0.3%)
Greatest absolute difference: 0.6601031422615051 at index (0, 1, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71059 / 28311552 (0.3%)
Greatest absolute difference: 0.9063564538955688 at index (14, 0, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71024 / 28311552 (0.3%)
Greatest absolute difference: 0.7093196511268616 at index (12, 2, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71072 / 28311552 (0.3%)
Greatest absolute difference: 0.6153253316879272 at index (6, 0, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71057 / 28311552 (0.3%)
Greatest absolute difference: 0.65338134765625 at index (5, 2, 0, 475) (up to 0.01 allowed) |
| repair API call #1 | PASS | 71.35s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 110.26s | rejected; AssertionError: Tensor-likes are not close!

Mismatched elements: 70950 / 28311552 (0.3%)
Greatest absolute difference: 0.6601031422615051 at index (0, 1, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71059 / 28311552 (0.3%)
Greatest absolute difference: 0.9063565731048584 at index (14, 0, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71024 / 28311552 (0.3%)
Greatest absolute difference: 0.709319531917572 at index (12, 2, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71072 / 28311552 (0.3%)
Greatest absolute difference: 0.6153253316879272 at index (6, 0, 0, 0) (up to 0.01 allowed)
Mismatched elements: 71057 / 28311552 (0.3%)
Greatest absolute difference: 0.6533814668655396 at index (5, 2, 0, 475) (up to 0.01 allowed) |
| repair API call #2 | PASS | 17.67s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 124.67s | best |
| profiling | PASS | 22.91s |  |
| artifacts | PASS | 1.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 314 --task-end 314
```
