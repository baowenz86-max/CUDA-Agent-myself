# CUDA Agent Workflow Report

- Task ID: `471`
- Status: **SUCCESS**
- Operators: `["torch.view_as_real", "nn.LogSoftmax", "F.group_norm"]`
- Started: `2026-11-10T16:07:29`
- Finished: `2026-11-10T16:12:31`
- Duration: `302.37s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.77s |  |
| generation #1 | PASS | 122.15s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 115.15s | best; initial_candidate |
| profiling | PASS | 42.61s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 471 --task-end 471
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
