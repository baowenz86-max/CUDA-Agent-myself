# CUDA Agent Workflow Report

- Task ID: `661`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.equal"]`
- Started: `2026-11-11T10:14:24`
- Finished: `2026-11-11T10:20:38`
- Duration: `374.68s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.72s |  |
| generation #1 | PASS | 65.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 206.59s | best; initial_candidate; torch.AcceleratorError: CUDA error: invalid argument
Search for `cudaErrorInvalidValue' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.

RuntimeError: Comparing
TensorLikePair(
    id=(), |
| repair API call #1 | PASS | 28.51s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 45.65s | best; correctness_success |
| profiling | PASS | 18.56s |  |
| artifacts | PASS | 5.78s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 661 --task-end 661
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
