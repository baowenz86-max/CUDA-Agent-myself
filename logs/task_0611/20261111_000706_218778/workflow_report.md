# CUDA Agent Workflow Report

- Task ID: `611`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.LayerNorm"]`
- Started: `2026-11-11T00:07:06`
- Finished: `2026-11-11T00:11:31`
- Duration: `265.26s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.98s |  |
| generation #1 | PASS | 53.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 47.73s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 63062 / 65536 (96.2%)
Greatest absolute difference: 20.716581344604492 at index (302, 19) (up to 0.01 allowed)
Mismatched elements: 63011 / 65536 (96.1%)
Greatest absolute difference: 25.998092651367188 at index (441, 102) (up to 0.01 allowed)
Mismatched elements: 63065 / 65536 (96.2%)
Greatest absolute difference: 21.86595916748047 at index (403, 28) (up to 0.01 allowed)
Mismatched elements: 63097 / 65536 (96.3%)
Greatest absolute difference: 21.76690673828125 at index (304, 111) (up to 0.01 allowed)
Mismatched elements: 63161 / 65536 (96.4%)
Greatest absolute difference: 24.026845932006836 at index (225, 66) (up to 0.01 allowed) |
| repair API call #1 | PASS | 25.02s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 81.91s | best; correctness_success |
| profiling | PASS | 46.47s |  |
| artifacts | PASS | 1.01s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 611 --task-end 611
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
