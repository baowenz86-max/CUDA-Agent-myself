# CUDA Agent Workflow Report

- Task ID: `711`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.PixelShuffle"]`
- Started: `2026-11-11T11:32:58`
- Finished: `2026-11-11T11:36:52`
- Duration: `233.91s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.24s |  |
| generation #1 | PASS | 66.60s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 122.78s | best; initial_candidate |
| profiling | PASS | 18.37s |  |
| artifacts | PASS | 1.85s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 711 --task-end 711
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
