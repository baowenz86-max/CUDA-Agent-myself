# CUDA Agent Workflow Report

- Task ID: `607`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.Unfold"]`
- Started: `2026-11-11T00:00:34`
- Finished: `2026-11-11T00:03:33`
- Duration: `178.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.80s |  |
| generation #1 | PASS | 92.27s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 54.83s | best; initial_candidate |
| profiling | PASS | 27.98s |  |
| artifacts | PASS | 0.95s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 607 --task-end 607
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
