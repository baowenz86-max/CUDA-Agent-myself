# CUDA Agent Workflow Report

- Task ID: `595`
- Status: **SUCCESS**
- Operators: `["torch.equal", "nn.AdaptiveAvgPool1d"]`
- Started: `2026-11-10T23:24:59`
- Finished: `2026-11-10T23:29:00`
- Duration: `241.55s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 33.20s |  |
| generation #1 | PASS | 37.37s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 94.05s | best; initial_candidate |
| profiling | PASS | 74.16s |  |
| artifacts | PASS | 2.48s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 595 --task-end 595
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
