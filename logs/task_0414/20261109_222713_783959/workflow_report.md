# CUDA Agent Workflow Report

- Task ID: `414`
- Status: **SUCCESS**
- Operators: `["F.interpolate"]`
- Started: `2026-11-09T22:27:13`
- Finished: `2026-11-09T22:30:47`
- Duration: `213.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.85s |  |
| generation #1 | PASS | 66.84s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 43.92s | best; initial_candidate |
| profiling | PASS | 22.14s |  |
| artifacts | PASS | 77.82s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 414 --task-end 414
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
