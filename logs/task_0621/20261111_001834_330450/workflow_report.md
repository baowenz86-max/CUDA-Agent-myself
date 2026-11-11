# CUDA Agent Workflow Report

- Task ID: `621`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.AvgPool2d"]`
- Started: `2026-11-11T00:18:34`
- Finished: `2026-11-11T00:23:44`
- Duration: `310.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.34s |  |
| generation #1 | PASS | 56.47s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 163.51s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 520608 / 26738688 (1.9%)
Greatest absolute difference: 1.0694506168365479 at index (40, 1, 511, 255) (up to 0.01 allowed)
Greatest absolute difference: 0.9177248477935791 at index (47, 1, 511, 255) (up to 0.01 allowed)
Greatest absolute difference: 0.95028156042099 at index (18, 1, 0, 255) (up to 0.01 allowed)
Greatest absolute difference: 1.0289030075073242 at index (3, 1, 0, 255) (up to 0.01 allowed)
Greatest absolute difference: 1.1017911434173584 at index (16, 2, 511, 255) (up to 0.01 allowed) |
| repair API call #1 | PASS | 36.45s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 11.84s | best; correctness_success |
| profiling | PASS | 18.10s |  |
| artifacts | PASS | 7.22s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 621 --task-end 621
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
