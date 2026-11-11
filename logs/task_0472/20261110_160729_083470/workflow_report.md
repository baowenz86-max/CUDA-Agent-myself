# CUDA Agent Workflow Report

- Task ID: `472`
- Status: **SUCCESS**
- Operators: `["torch.softmax", "torch.relu"]`
- Started: `2026-11-10T16:07:29`
- Finished: `2026-11-10T16:11:22`
- Duration: `233.00s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.29s |  |
| generation #1 | PASS | 88.02s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 45.65s | best; initial_candidate |
| profiling | PASS | 18.39s |  |
| artifacts | PASS | 75.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 472 --task-end 472
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
