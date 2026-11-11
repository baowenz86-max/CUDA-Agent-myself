# CUDA Agent Workflow Report

- Task ID: `393`
- Status: **SUCCESS**
- Operators: `["nn.NLLLoss"]`
- Started: `2026-11-09T21:54:03`
- Finished: `2026-11-09T22:00:26`
- Duration: `382.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.57s |  |
| generation #1 | PASS | 70.30s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 49.56s | best; initial_candidate; torch.AcceleratorError: CUDA error: invalid argument
Search for `cudaErrorInvalidValue' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.

RuntimeError: Comparing
TensorLikePair(
    id=(), |
| repair API call #1 | PASS | 17.02s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 203.70s | best; correctness_success |
| profiling | PASS | 34.77s |  |
| artifacts | PASS | 3.58s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 393 --task-end 393
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
