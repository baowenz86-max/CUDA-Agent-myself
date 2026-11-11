# CUDA Agent Workflow Report

- Task ID: `480`
- Status: **SUCCESS**
- Operators: `["torch.relu", "F.softplus"]`
- Started: `2026-11-10T16:15:43`
- Finished: `2026-11-10T16:18:58`
- Duration: `194.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.62s |  |
| generation #1 | PASS | 53.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 46.15s | best; initial_candidate |
| profiling | PASS | 26.29s |  |
| artifacts | PASS | 57.96s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 480 --task-end 480
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
