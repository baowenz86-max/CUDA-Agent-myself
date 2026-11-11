# CUDA Agent Workflow Report

- Task ID: `593`
- Status: **FAILED**
- Operators: `["torch.abs", "F.layer_norm"]`
- Started: `2026-11-10T20:40:18`
- Finished: `2026-11-10T20:41:21`
- Duration: `63.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.36s |  |
| generation #1 | FAIL | 47.06s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 593 --task-end 593
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
