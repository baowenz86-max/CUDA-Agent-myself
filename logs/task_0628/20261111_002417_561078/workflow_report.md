# CUDA Agent Workflow Report

- Task ID: `628`
- Status: **FAILED**
- Operators: `["torch.isnan", "nn.MultiheadAttention"]`
- Started: `2026-11-11T00:24:17`
- Finished: `2026-11-11T00:25:18`
- Duration: `61.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.65s |  |
| generation #1 | FAIL | 58.61s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 628 --task-end 628
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
