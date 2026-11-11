# CUDA Agent Workflow Report

- Task ID: `732`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.acos"]`
- Started: `2026-11-11T13:06:10`
- Finished: `2026-11-11T13:12:04`
- Duration: `354.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 27.97s |  |
| generation #1 | PASS | 39.61s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 174.74s | best; initial_candidate |
| profiling | PASS | 110.32s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 732 --task-end 732
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
