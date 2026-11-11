# CUDA Agent Workflow Report

- Task ID: `300`
- Status: **SUCCESS**
- Operators: `["nn.LayerNorm", "nn.functional"]`
- Started: `2026-11-07T14:45:29`
- Finished: `2026-11-07T14:53:55`
- Duration: `505.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.04s |  |
| generation #1 | PASS | 42.73s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 282.36s | best; TypeError: layer_norm_forward(): incompatible function arguments. The following argument types are supported:
    1. (input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, normalized_shape: typing.SupportsInt \| typing.SupportsIndex, eps: typing.SupportsFloat \| typing.SupportsIndex) -> torch.Tensor

Invoked with: tensor([[[-0.0113, -0.1229, -1.4293,  ...,  0.7785,  1.5198,  1.1811], |
| repair API call #1 | PASS | 10.15s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 54.40s | best |
| profiling | PASS | 98.23s |  |
| artifacts | PASS | 1.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 300 --task-end 300
```
