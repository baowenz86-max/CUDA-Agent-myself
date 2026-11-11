# CUDA Agent Workflow Report

- Task ID: `741`
- Status: **SUCCESS**
- Operators: `["F.softmax", "nn.ConstantPad1d"]`
- Started: `2026-11-11T13:19:29`
- Finished: `2026-11-11T13:22:52`
- Duration: `203.27s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.13s |  |
| generation #1 | PASS | 42.31s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 30.66s | best; initial_candidate; TypeError: pad_forward(): incompatible function arguments. The following argument types are supported:
    1. (x: torch.Tensor, padding: typing.SupportsInt \| typing.SupportsIndex, value: typing.SupportsFloat \| typing.SupportsIndex) -> torch.Tensor

Invoked with: tensor([[[0.0018, 0.0014, 0.0109,  ..., 0.0076, 0.0118, 0.0045], |
| repair API call #1 | PASS | 10.86s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 82.44s | best; correctness_success |
| profiling | PASS | 15.57s |  |
| artifacts | PASS | 1.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 741 --task-end 741
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
