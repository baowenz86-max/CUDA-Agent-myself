# CUDA Agent Workflow Report

- Task ID: `811`
- Status: **FAILED**
- Operators: `["nn.Upsample", "torch.softmax"]`
- Started: `2026-11-11T15:46:53`
- Finished: `2026-11-11T15:48:48`
- Duration: `115.33s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 57.50s |  |
| generation #1 | FAIL | 57.77s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 811 --task-end 811
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
