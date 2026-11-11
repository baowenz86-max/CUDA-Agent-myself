# CUDA Agent Workflow Report

- Task ID: `666`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.triu"]`
- Started: `2026-11-11T10:27:13`
- Finished: `2026-11-11T10:30:31`
- Duration: `198.13s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.42s |  |
| generation #1 | PASS | 68.73s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 76.76s | best; initial_candidate |
| profiling | PASS | 16.40s |  |
| artifacts | PASS | 1.74s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 666 --task-end 666
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
