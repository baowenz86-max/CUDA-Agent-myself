# CUDA Agent Workflow Report

- Task ID: `387`
- Status: **FAILED**
- Operators: `["F.ctc_loss"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:38:04`
- Duration: `163.85s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 105.43s |  |
| generation #1 | FAIL | 58.10s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 387 --task-end 387
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
