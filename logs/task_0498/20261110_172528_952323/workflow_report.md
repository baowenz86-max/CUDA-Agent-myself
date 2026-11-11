# CUDA Agent Workflow Report

- Task ID: `498`
- Status: **SUCCESS**
- Operators: `["F.gelu", "torch.prod"]`
- Started: `2026-11-10T17:25:28`
- Finished: `2026-11-10T17:28:17`
- Duration: `168.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.26s |  |
| generation #1 | PASS | 57.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 58.11s | best; initial_candidate |
| profiling | PASS | 19.36s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 498 --task-end 498
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
