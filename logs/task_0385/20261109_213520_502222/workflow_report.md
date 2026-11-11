# CUDA Agent Workflow Report

- Task ID: `385`
- Status: **SUCCESS**
- Operators: `["torch.clamp", "nn.LayerNorm"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:41:33`
- Duration: `372.55s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 72.34s |  |
| generation #1 | PASS | 106.24s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 126.21s | best; initial_candidate |
| profiling | PASS | 64.17s |  |
| artifacts | PASS | 3.14s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 385 --task-end 385
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
