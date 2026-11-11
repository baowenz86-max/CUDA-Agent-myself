# CUDA Agent Workflow Report

- Task ID: `749`
- Status: **FAILED**
- Operators: `["torch.cat", "torch.unbind"]`
- Started: `2026-11-11T13:26:35`
- Finished: `2026-11-11T13:26:51`
- Duration: `16.23s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.53s |  |
| generation #1 | FAIL | 8.63s | Single Python → CUDA → C++ binding generation |

## Failure Summary

Initial three-stage generation failed; no regeneration or repair.

## Reproduction

```bash
python main.py --task-start 749 --task-end 749
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
