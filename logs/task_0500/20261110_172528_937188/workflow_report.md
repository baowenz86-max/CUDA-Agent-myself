# CUDA Agent Workflow Report

- Task ID: `500`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "torch.bmm"]`
- Started: `2026-11-10T17:25:28`
- Finished: `2026-11-10T17:26:59`
- Duration: `91.01s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.00s |  |
| generation #1 | PASS | 47.06s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 26.49s | best; initial_candidate |
| profiling | PASS | 9.83s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 500 --task-end 500
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
