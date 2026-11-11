# CUDA Agent Workflow Report

- Task ID: `532`
- Status: **FAILED**
- Operators: `["nn.Embedding", "nn.KLDivLoss"]`
- Started: `2026-11-10T18:33:10`
- Finished: `2026-11-10T18:34:29`
- Duration: `79.22s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.28s |  |
| generation #1 | FAIL | 72.87s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 532 --task-end 532
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
