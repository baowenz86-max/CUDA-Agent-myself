# CUDA Agent Workflow Report

- Task ID: `824`
- Status: **FAILED**
- Operators: `["nn.BatchNorm2d", "nn.MaxPool1d"]`
- Started: `2026-11-11T15:32:34`
- Finished: `2026-11-11T15:35:00`
- Duration: `146.38s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 51.76s |  |
| generation #1 | FAIL | 94.55s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 824 --task-end 824
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
