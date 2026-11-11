# CUDA Agent Workflow Report

- Task ID: `566`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.gt"]`
- Started: `2026-11-10T19:57:55`
- Finished: `2026-11-10T20:02:03`
- Duration: `248.39s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.28s |  |
| generation #1 | PASS | 43.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 153.71s | best; initial_candidate |
| profiling | PASS | 32.47s |  |
| artifacts | PASS | 1.56s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 566 --task-end 566
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
