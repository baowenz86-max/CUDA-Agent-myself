# CUDA Agent Workflow Report

- Task ID: `735`
- Status: **SUCCESS**
- Operators: `["F.elu", "torch.relu"]`
- Started: `2026-11-11T13:06:10`
- Finished: `2026-11-11T13:12:16`
- Duration: `366.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 61.61s |  |
| generation #1 | PASS | 32.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 213.26s | best; initial_candidate |
| profiling | PASS | 57.16s |  |
| artifacts | PASS | 1.58s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 735 --task-end 735
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
