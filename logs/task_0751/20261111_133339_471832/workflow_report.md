# CUDA Agent Workflow Report

- Task ID: `751`
- Status: **SUCCESS**
- Operators: `["nn.Linear", "torch.minimum"]`
- Started: `2026-11-11T13:33:39`
- Finished: `2026-11-11T13:37:03`
- Duration: `204.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.86s |  |
| generation #1 | PASS | 45.74s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 93.70s | best; initial_candidate |
| profiling | PASS | 24.45s |  |
| artifacts | PASS | 26.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 751 --task-end 751
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
