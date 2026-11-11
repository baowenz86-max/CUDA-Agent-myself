# CUDA Agent Workflow Report

- Task ID: `404`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.where"]`
- Started: `2026-11-09T22:13:21`
- Finished: `2026-11-09T22:16:20`
- Duration: `179.81s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.08s |  |
| generation #1 | PASS | 46.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 77.70s | best; initial_candidate |
| profiling | PASS | 35.05s |  |
| artifacts | PASS | 2.37s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 404 --task-end 404
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
