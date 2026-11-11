# CUDA Agent Workflow Report

- Task ID: `340`
- Status: **SUCCESS**
- Operators: `["torch.selu", "torch.cumsum"]`
- Started: `2026-11-07T16:17:25`
- Finished: `2026-11-07T16:23:45`
- Duration: `380.30s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 36.28s |  |
| generation #1 | PASS | 45.98s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 59.11s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0340/build/forced_compile/cuda_extension.so: undefined symbol: selu_scale
[TIME] Compilation took 25.98s |
| repair API call #1 | PASS | 19.19s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 77.14s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 33300 / 100000 (33.3%)
Greatest absolute difference: 0.9979820251464844 at index (18843, 2) (up to 0.01 allowed)
Mismatched elements: 40930 / 100000 (40.9%)
Greatest absolute difference: 0.9992785453796387 at index (367, 4) (up to 0.01 allowed)
Mismatched elements: 32817 / 100000 (32.8%)
Greatest absolute difference: 0.997406005859375 at index (15124, 2) (up to 0.01 allowed)
Mismatched elements: 36243 / 100000 (36.2%)
Greatest absolute difference: 0.9993171691894531 at index (11139, 1) (up to 0.01 allowed)
Mismatched elements: 34657 / 100000 (34.7%)
Greatest absolute difference: 0.9997177124023438 at index (11842, 2) (up to 0.01 allowed) |
| repair API call #2 | PASS | 30.12s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 45.49s | best |
| profiling | PASS | 14.77s |  |
| artifacts | PASS | 51.83s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 340 --task-end 340
```
