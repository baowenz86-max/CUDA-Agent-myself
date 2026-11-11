# CUDA Agent Workflow Report

- Task ID: `704`
- Status: **FAILED**
- Operators: `["F.softmax", "nn.MSELoss"]`
- Started: `2026-11-11T11:16:26`
- Finished: `2026-11-11T11:17:48`
- Duration: `82.57s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.14s |  |
| generation #1 | FAIL | 37.38s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 704 --task-end 704
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
