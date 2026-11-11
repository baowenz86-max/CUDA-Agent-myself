# CUDA Agent Workflow Report

- Task ID: `822`
- Status: **FAILED**
- Operators: `["nn.AdaptiveLogSoftmaxWithLoss", "torch.relu"]`
- Started: `2026-11-11T17:49:13`
- Finished: `2026-11-11T17:53:14`
- Duration: `240.94s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.44s |  |
| generation #1 | FAIL | 226.34s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 822 --task-end 822
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
