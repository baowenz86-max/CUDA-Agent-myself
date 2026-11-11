# CUDA Agent Workflow Report

- Task ID: `427`
- Status: **FAILED**
- Operators: `["torch.max", "nn.GroupNorm"]`
- Started: `2026-11-09T23:34:35`
- Finished: `2026-11-09T23:36:11`
- Duration: `95.74s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.24s |  |
| generation #1 | FAIL | 64.45s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 427 --task-end 427
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
