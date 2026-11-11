# CUDA Agent Workflow Report

- Task ID: `715`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.PReLU"]`
- Started: `2026-11-11T11:32:58`
- Finished: `2026-11-11T11:37:48`
- Duration: `290.18s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.92s |  |
| generation #1 | PASS | 65.66s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 142.93s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 131853956 / 134217728 (98.2%)
Greatest absolute difference: 5.5128631591796875 at index (11, 35, 201, 127) (up to 0.01 allowed)
Mismatched elements: 131855767 / 134217728 (98.2%)
Greatest absolute difference: 5.418915271759033 at index (1, 0, 58, 13) (up to 0.01 allowed)
Mismatched elements: 131852735 / 134217728 (98.2%)
Greatest absolute difference: 5.41392707824707 at index (4, 41, 21, 321) (up to 0.01 allowed)
Mismatched elements: 131854087 / 134217728 (98.2%)
Greatest absolute difference: 5.5551276206970215 at index (7, 24, 42, 320) (up to 0.01 allowed)
Mismatched elements: 131851016 / 134217728 (98.2%)
Greatest absolute difference: 5.402670383453369 at index (14, 55, 121, 475) (up to 0.01 allowed) |
| repair API call #1 | PASS | 21.76s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 38.10s | best; correctness_success |
| profiling | PASS | 10.56s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 715 --task-end 715
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
