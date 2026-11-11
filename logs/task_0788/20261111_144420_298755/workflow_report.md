# CUDA Agent Workflow Report

- Task ID: `788`
- Status: **SUCCESS**
- Operators: `["F.softmax", "F.softmax"]`
- Started: `2026-11-11T14:44:20`
- Finished: `2026-11-11T14:54:51`
- Duration: `631.07s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.85s |  |
| generation #1 | PASS | 43.92s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 42.71s | best; initial_candidate; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (x: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[[ 0.2197, -0.4136,  0.2948,  ...,  1.8991,  1.7482, -1.9387], |
| repair API call #1 | PASS | 14.95s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 182.59s | rejected; tie_keep_previous; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (x: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[[ 0.2197, -0.4136,  0.2948,  ...,  1.8991,  1.7482, -1.9387], |
| repair API call #2 | PASS | 128.17s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 29.77s | rejected; tie_keep_previous; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (x: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[[ 0.2197, -0.4136,  0.2948,  ...,  1.8991,  1.7482, -1.9387], |
| repair API call #3 | PASS | 17.54s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | PASS | 106.55s | best; correctness_success |
| profiling | PASS | 36.22s |  |
| artifacts | PASS | 6.96s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 788 --task-end 788
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
