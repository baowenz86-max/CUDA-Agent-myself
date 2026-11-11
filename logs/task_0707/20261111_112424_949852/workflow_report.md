# CUDA Agent Workflow Report

- Task ID: `707`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.embedding"]`
- Started: `2026-11-11T11:24:24`
- Finished: `2026-11-11T11:26:07`
- Duration: `102.44s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.45s |  |
| generation #1 | PASS | 58.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 28.26s | best; initial_candidate |
| profiling | PASS | 11.10s |  |
| artifacts | PASS | 1.66s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 707 --task-end 707
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
