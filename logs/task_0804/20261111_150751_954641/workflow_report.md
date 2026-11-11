# CUDA Agent Workflow Report

- Task ID: `804`
- Status: **SUCCESS**
- Operators: `["torch.stack", "torch.stack"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:13:39`
- Duration: `347.16s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 48.32s |  |
| generation #1 | PASS | 67.62s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 183.49s | best; initial_candidate |
| profiling | PASS | 46.12s |  |
| artifacts | PASS | 1.54s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 804 --task-end 804
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
