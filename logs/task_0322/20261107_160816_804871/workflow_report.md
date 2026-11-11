# CUDA Agent Workflow Report

- Task ID: `322`
- Status: **SUCCESS**
- Operators: `["F.native_channel_shuffle"]`
- Started: `2026-11-07T16:08:16`
- Finished: `2026-11-07T16:15:01`
- Duration: `404.54s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.18s |  |
| generation #1 | PASS | 25.65s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 73.56s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 249436546 / 268435456 (92.9%)
Greatest absolute difference: 8.372780799865723 at index (190, 15, 88, 27) (up to 0.01 allowed)
Mismatched elements: 249435221 / 268435456 (92.9%)
Greatest absolute difference: 8.106589317321777 at index (438, 1, 36, 35) (up to 0.01 allowed)
Mismatched elements: 249437950 / 268435456 (92.9%)
Greatest absolute difference: 8.431464195251465 at index (360, 29, 7, 12) (up to 0.01 allowed)
Mismatched elements: 249440077 / 268435456 (92.9%)
Greatest absolute difference: 8.497217178344727 at index (471, 16, 123, 13) (up to 0.01 allowed)
Mismatched elements: 249433432 / 268435456 (92.9%)
Greatest absolute difference: 8.552587509155273 at index (269, 22, 55, 16) (up to 0.01 allowed) |
| repair API call #1 | PASS | 11.13s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 235.64s | best |
| profiling | PASS | 23.46s |  |
| artifacts | PASS | 29.51s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 322 --task-end 322
```
