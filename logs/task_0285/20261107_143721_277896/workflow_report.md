# CUDA Agent Workflow Report

- Task ID: `285`
- Status: **SUCCESS**
- Operators: `["torch.pixel_unshuffle"]`
- Started: `2026-11-07T14:37:21`
- Finished: `2026-11-07T14:44:24`
- Duration: `422.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 41.10s |  |
| generation #1 | PASS | 37.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 203.79s | best; AssertionError: Tensor-likes are not close!

Mismatched elements: 174487350 / 201326592 (86.7%)
Greatest absolute difference: 8.248944282531738 at index (51, 7, 111, 24) (up to 0.01 allowed)
Mismatched elements: 174474478 / 201326592 (86.7%)
Greatest absolute difference: 9.07649040222168 at index (57, 7, 376, 487) (up to 0.01 allowed)
Mismatched elements: 174491483 / 201326592 (86.7%)
Greatest absolute difference: 8.035128593444824 at index (12, 3, 486, 417) (up to 0.01 allowed)
Mismatched elements: 174445024 / 201326592 (86.6%)
Greatest absolute difference: 8.46258544921875 at index (22, 9, 307, 336) (up to 0.01 allowed)
Mismatched elements: 174467386 / 201326592 (86.7%)
Greatest absolute difference: 8.351767539978027 at index (29, 11, 18, 86) (up to 0.01 allowed) |
| repair API call #1 | PASS | 28.38s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 67.68s | best |
| profiling | PASS | 41.96s |  |
| artifacts | PASS | 1.85s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 285 --task-end 285
```
