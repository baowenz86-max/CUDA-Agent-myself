# CUDA Agent Workflow Report

- Task ID: `547`
- Status: **SUCCESS**
- Operators: `["torch.triu", "torch.abs"]`
- Started: `2026-11-10T19:28:46`
- Finished: `2026-11-10T19:31:59`
- Duration: `193.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.50s |  |
| generation #1 | PASS | 69.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 95.36s | best; initial_candidate |
| profiling | PASS | 11.54s |  |
| artifacts | PASS | 0.99s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 547 --task-end 547
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
