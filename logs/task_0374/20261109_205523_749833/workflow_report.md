# CUDA Agent Workflow Report

- Task ID: `374`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.bucketize"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:03:58`
- Duration: `514.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.77s |  |
| generation #1 | PASS | 87.49s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 329.09s | best; initial_candidate |
| profiling | PASS | 50.40s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 374 --task-end 374
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
