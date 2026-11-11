# CUDA Agent Workflow Report

- Task ID: `833`
- Status: **SUCCESS**
- Operators: `["torch.triu", "nn.BatchNorm2d"]`
- Started: `2026-11-11T17:58:22`
- Finished: `2026-11-11T18:01:27`
- Duration: `185.18s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 50.05s |  |
| generation #1 | PASS | 56.32s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 61.49s | best; initial_candidate |
| profiling | PASS | 15.35s |  |
| artifacts | PASS | 1.91s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 833 --task-end 833
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
