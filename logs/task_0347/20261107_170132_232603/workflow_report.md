# CUDA Agent Workflow Report

- Task ID: `347`
- Status: **SUCCESS**
- Operators: `["nn.Unfold"]`
- Started: `2026-11-07T17:01:32`
- Finished: `2026-11-07T17:05:44`
- Duration: `251.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.08s |  |
| generation #1 | PASS | 48.65s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.04s | best; forward: torch.ops.cuda_extension.unfold is incompatible with the pybind cuda_extension module
debug_forward: torch.ops.cuda_extension.unfold is incompatible with the pybind cuda_extension module |
| repair API call #1 | PASS | 5.43s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 58.11s | best |
| profiling | PASS | 16.80s |  |
| artifacts | PASS | 114.73s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 347 --task-end 347
```
