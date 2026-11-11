# CUDA Agent Workflow Report

- Task ID: `467`
- Status: **SUCCESS**
- Operators: `["torch.triu", "F.hardtanh"]`
- Started: `2026-11-10T16:01:31`
- Finished: `2026-11-10T16:07:27`
- Duration: `356.71s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.16s |  |
| generation #1 | PASS | 77.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 106.48s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 103017671 / 104857600 (98.2%)
Greatest absolute difference: 3.2189128398895264 at index (91453, 617) (up to 0.01 allowed)
Mismatched elements: 103019198 / 104857600 (98.2%)
Greatest absolute difference: 3.2391371726989746 at index (96753, 992) (up to 0.01 allowed)
Mismatched elements: 103017581 / 104857600 (98.2%)
Greatest absolute difference: 3.187222480773926 at index (25736, 637) (up to 0.01 allowed)
Mismatched elements: 103016368 / 104857600 (98.2%)
Greatest absolute difference: 3.3748860359191895 at index (38470, 277) (up to 0.01 allowed)
Mismatched elements: 103017356 / 104857600 (98.2%)
Greatest absolute difference: 3.4370808601379395 at index (81145, 749) (up to 0.01 allowed) |
| repair API call #1 | PASS | 30.27s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 92.81s | best; correctness_success |
| profiling | PASS | 31.42s |  |
| artifacts | PASS | 1.00s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 467 --task-end 467
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
