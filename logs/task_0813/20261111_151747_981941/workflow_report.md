# CUDA Agent Workflow Report

- Task ID: `813`
- Status: **FAILED**
- Operators: `["torch.ceil", "nn.ReflectionPad2d"]`
- Started: `2026-11-11T15:17:47`
- Finished: `2026-11-11T15:19:09`
- Duration: `81.55s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.58s |  |
| generation #1 | FAIL | 48.91s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 813 --task-end 813
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
