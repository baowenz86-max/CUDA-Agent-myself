# CUDA Agent Workflow Report

- Task ID: `572`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.AdaptiveAvgPool2d"]`
- Started: `2026-11-10T20:02:15`
- Finished: `2026-11-10T20:05:47`
- Duration: `212.44s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.47s |  |
| generation #1 | PASS | 51.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 107.26s | best; initial_candidate |
| profiling | PASS | 48.64s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 572 --task-end 572
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
