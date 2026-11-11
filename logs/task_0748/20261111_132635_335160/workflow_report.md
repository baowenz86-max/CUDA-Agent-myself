# CUDA Agent Workflow Report

- Task ID: `748`
- Status: **FAILED**
- Operators: `["F.softmax", "torch.unsqueeze"]`
- Started: `2026-11-11T13:26:35`
- Finished: `2026-11-11T13:33:38`
- Duration: `423.01s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.89s |  |
| generation #1 | PASS | 69.69s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 96.92s | best; initial_candidate; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[-0.6736,  1.9036,  0.4535,  ..., -1.3385,  0.6404, -1.3675], |
| repair API call #1 | PASS | 23.85s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 26.08s | rejected; tie_keep_previous; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[-0.6736,  1.9036,  0.4535,  ..., -1.3385,  0.6404, -1.3675], |
| repair API call #2 | PASS | 16.80s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 82.05s | rejected; tie_keep_previous; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[-0.6736,  1.9036,  0.4535,  ..., -1.3385,  0.6404, -1.3675], |
| repair API call #3 | FAIL | 40.22s | Code repairs used: 2/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #4 | PASS | 23.50s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 26.71s | rejected; tie_keep_previous; TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, dim: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[-0.6736,  1.9036,  0.4535,  ..., -1.3385,  0.6404, -1.3675], |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"kind": "runtime", "message": "TypeError: custom_softmax(): incompatible function arguments. The following argument types are supported:\n    1. (input: at::Tensor, dim: typing.SupportsInt | typing.SupportsIndex) -> at::Tensor\n\nInvoked with: tensor([[-0.6736,  1.9036,  0.4535,  ..., -1.3385,  0.6404, -1.3675],\n        [ 0.4749,  0.6525,  0.2502,  ...,  0.5365,  0.6098, -0.3745],\n        [ 0.6553,  0.4633, -1.2850,  ...,  1.3558, -0.7848,  0.2082],\n        ...,\n        [-1.7163,  0.8548,  0.1919,  ..., -0.2962, -0.1468,  1.6110],\n        [ 0.8814,  1.4569,  0.7707,  ...,  0.2777, -0.0509, -0.4288],\n        [ 0.3834, -1.6875, -0.1010,  ...,  2.4780,  0.4775,  0.4940]],\n       device='cuda:0'), 0"}

## Reproduction

```bash
python main.py --task-start 748 --task-end 748
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
