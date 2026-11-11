# CUDA Agent Workflow Report

- Task ID: `493`
- Status: **SUCCESS**
- Operators: `["F.hardsigmoid", "nn.MaxPool2d"]`
- Started: `2026-11-10T17:21:50`
- Finished: `2026-11-10T17:25:27`
- Duration: `217.21s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.63s |  |
| generation #1 | PASS | 54.60s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 84.05s | best; initial_candidate; TypeError: max_pool2d_forward(): incompatible function arguments. The following argument types are supported:
    1. (x: torch.Tensor, kernel_size: typing.SupportsInt \| typing.SupportsIndex, stride: typing.SupportsInt \| typing.SupportsIndex, padding: typing.SupportsInt \| typing.SupportsIndex) -> torch.Tensor

Invoked with: tensor([[[[ 1.4212,  0.0000,  2.1354,  ...,  0.0000,  0.0000,  0.7300], |
| repair API call #1 | PASS | 15.96s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 17.23s | best; correctness_success |
| profiling | PASS | 11.12s |  |
| artifacts | PASS | 0.98s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 493 --task-end 493
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
