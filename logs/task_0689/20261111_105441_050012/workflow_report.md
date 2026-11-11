# CUDA Agent Workflow Report

- Task ID: `689`
- Status: **SUCCESS**
- Operators: `["torch.mean", "F.cosine_similarity"]`
- Started: `2026-11-11T10:54:41`
- Finished: `2026-11-11T10:57:13`
- Duration: `152.07s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.32s |  |
| generation #1 | PASS | 48.10s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 51.85s | best; initial_candidate |
| profiling | PASS | 23.55s |  |
| artifacts | PASS | 24.18s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 689 --task-end 689
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
