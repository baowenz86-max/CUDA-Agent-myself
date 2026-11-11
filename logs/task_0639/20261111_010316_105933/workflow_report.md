# CUDA Agent Workflow Report

- Task ID: `639`
- Status: **SUCCESS**
- Operators: `["torch.max", "nn.LayerNorm"]`
- Started: `2026-11-11T01:03:16`
- Finished: `2026-11-11T01:09:05`
- Duration: `349.78s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 61.11s |  |
| generation #1 | PASS | 60.76s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 194.65s | best; initial_candidate |
| profiling | PASS | 31.12s |  |
| artifacts | PASS | 1.88s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 639 --task-end 639
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
