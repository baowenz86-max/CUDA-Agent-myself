# CUDA Agent Workflow Report

- Task ID: `469`
- Status: **FAILED**
- Operators: `["torch.fmod", "nn.ReflectionPad2d"]`
- Started: `2026-11-10T16:01:31`
- Finished: `2026-11-10T16:02:32`
- Duration: `61.16s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.00s |  |
| generation #1 | FAIL | 57.10s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 469 --task-end 469
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
