# CUDA Agent Workflow Report

- Task ID: `589`
- Status: **SUCCESS**
- Operators: `["torch.tril", "F.elu"]`
- Started: `2026-11-10T20:36:03`
- Finished: `2026-11-10T20:39:22`
- Duration: `199.03s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 9.00s |  |
| generation #1 | PASS | 57.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 70.69s | best; initial_candidate |
| profiling | PASS | 39.13s |  |
| artifacts | PASS | 22.21s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 589 --task-end 589
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
