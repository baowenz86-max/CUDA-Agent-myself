# CUDA Agent Workflow Report

- Task ID: `603`
- Status: **SUCCESS**
- Operators: `["F.gelu", "torch.mean"]`
- Started: `2026-11-10T23:40:18`
- Finished: `2026-11-10T23:43:29`
- Duration: `190.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.61s |  |
| generation #1 | PASS | 45.84s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 52.67s | best; initial_candidate |
| profiling | PASS | 27.16s |  |
| artifacts | PASS | 54.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 603 --task-end 603
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
