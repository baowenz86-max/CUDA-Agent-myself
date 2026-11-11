# CUDA Agent Workflow Report

- Task ID: `786`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.layer_norm"]`
- Started: `2026-11-11T14:44:20`
- Finished: `2026-11-11T14:54:26`
- Duration: `606.19s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.53s |  |
| generation #1 | PASS | 78.53s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 154.01s | best; initial_candidate; TypeError: custom_layer_norm(): incompatible function arguments. The following argument types are supported:
    1. (x: torch.Tensor, normalized_shape: torch.Tensor) -> torch.Tensor

Invoked with: tensor([[[[ 0.0423, -0.0453, -0.0048,  ..., -0.0025,  0.0127,  0.0385], |
| repair API call #1 | PASS | 22.76s | Code repairs used: 1/3; base candidate #1 |
| repair #1: sources candidate #2 | FAIL | 0.06s | rejected; missing_normal_observations; custom_layer_norm.normalized_shape: C++ type/nullable differs |
| repair API call #2 | PASS | 9.00s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 118.19s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 102832 / 1048576 (9.8%)
Greatest absolute difference: 5.561674118041992 at index (1006, 0, 27, 0) (up to 0.01 allowed)
Mismatched elements: 110752 / 1048576 (10.6%)
Greatest absolute difference: 6.238922119140625 at index (959, 0, 4, 0) (up to 0.01 allowed)
Mismatched elements: 104006 / 1048576 (9.9%)
Greatest absolute difference: 6.490073204040527 at index (256, 0, 17, 0) (up to 0.01 allowed)
Mismatched elements: 107216 / 1048576 (10.2%)
Greatest absolute difference: 6.200990200042725 at index (762, 0, 12, 0) (up to 0.01 allowed)
Mismatched elements: 113150 / 1048576 (10.8%)
Greatest absolute difference: 5.734269142150879 at index (977, 0, 29, 0) (up to 0.01 allowed) |
| repair API call #3 | PASS | 30.68s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | PASS | 120.90s | best; correctness_success |
| profiling | PASS | 23.46s |  |
| artifacts | PASS | 1.86s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 786 --task-end 786
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
