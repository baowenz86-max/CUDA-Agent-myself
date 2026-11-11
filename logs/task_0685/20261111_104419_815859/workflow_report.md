# CUDA Agent Workflow Report

- Task ID: `685`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.group_norm"]`
- Started: `2026-11-11T10:44:19`
- Finished: `2026-11-11T10:48:14`
- Duration: `234.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.74s |  |
| generation #1 | PASS | 62.41s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 113.36s | best; initial_candidate |
| profiling | PASS | 21.11s |  |
| artifacts | PASS | 1.74s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 685 --task-end 685
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
