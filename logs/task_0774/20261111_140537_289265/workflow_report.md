# CUDA Agent Workflow Report

- Task ID: `774`
- Status: **FAILED**
- Operators: `["F.softmax", "torch.einsum"]`
- Started: `2026-11-11T14:05:37`
- Finished: `2026-11-11T14:06:50`
- Duration: `73.39s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.72s |  |
| generation #1 | FAIL | 45.62s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 774 --task-end 774
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
