# CUDA Agent Workflow Report

- Task ID: `767`
- Status: **SUCCESS**
- Operators: `["F.softmax", "torch.exp"]`
- Started: `2026-11-11T13:50:45`
- Finished: `2026-11-11T13:53:55`
- Duration: `189.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.12s |  |
| generation #1 | PASS | 58.84s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 102.72s | best; initial_candidate |
| profiling | PASS | 12.37s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 767 --task-end 767
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
