# CUDA Agent Workflow Report

- Task ID: `443`
- Status: **SUCCESS**
- Operators: `["F.gelu", "torch.cumsum"]`
- Started: `2026-11-09T23:51:07`
- Finished: `2026-11-09T23:54:08`
- Duration: `181.51s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.88s |  |
| generation #1 | PASS | 33.99s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 52.87s | best; initial_candidate |
| profiling | PASS | 33.32s |  |
| artifacts | PASS | 56.38s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 443 --task-end 443
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
