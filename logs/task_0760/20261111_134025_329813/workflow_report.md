# CUDA Agent Workflow Report

- Task ID: `760`
- Status: **FAILED**
- Operators: `["F.softmax", "nn.LogSoftmax"]`
- Started: `2026-11-11T13:40:25`
- Finished: `2026-11-11T13:42:14`
- Duration: `108.76s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 49.80s |  |
| generation #1 | FAIL | 58.91s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 760 --task-end 760
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
