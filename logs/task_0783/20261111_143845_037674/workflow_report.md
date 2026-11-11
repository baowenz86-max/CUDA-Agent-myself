# CUDA Agent Workflow Report

- Task ID: `783`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.masked_fill"]`
- Started: `2026-11-11T14:38:45`
- Finished: `2026-11-11T14:42:44`
- Duration: `239.15s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.13s |  |
| generation #1 | PASS | 66.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 96.92s | best; initial_candidate |
| profiling | PASS | 53.39s |  |
| artifacts | PASS | 1.54s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 783 --task-end 783
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
