# CUDA Agent Workflow Report

- Task ID: `642`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.nan_to_num"]`
- Started: `2026-11-11T01:11:28`
- Finished: `2026-11-11T01:18:29`
- Duration: `420.73s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 58.94s |  |
| generation #1 | PASS | 25.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 256.87s | best; initial_candidate |
| profiling | PASS | 68.15s |  |
| artifacts | PASS | 10.81s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 642 --task-end 642
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
