# CUDA Agent Workflow Report

- Task ID: `463`
- Status: **SUCCESS**
- Operators: `["torch.minimum", "nn.ReflectionPad2d"]`
- Started: `2026-11-10T15:30:31`
- Finished: `2026-11-10T15:35:26`
- Duration: `294.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.61s |  |
| generation #1 | PASS | 63.78s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 159.88s | best; initial_candidate |
| profiling | PASS | 54.39s |  |
| artifacts | PASS | 1.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 463 --task-end 463
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
