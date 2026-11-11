# CUDA Agent Workflow Report

- Task ID: `495`
- Status: **FAILED**
- Operators: `["nn.MaxPool2d"]`
- Started: `2026-11-10T17:21:50`
- Finished: `2026-11-10T17:22:41`
- Duration: `51.18s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.70s |  |
| generation #1 | FAIL | 26.43s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 495 --task-end 495
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
