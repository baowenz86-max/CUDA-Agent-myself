# CUDA Agent Workflow Report

- Task ID: `603`
- Status: **SUCCESS**
- Operators: `["F.gelu", "torch.mean"]`
- Started: `2026-11-10T23:53:06`
- Finished: `2026-11-10T23:55:48`
- Duration: `161.67s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.20s |  |
| generation #1 | PASS | 50.17s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 82.06s | best; initial_candidate |
| profiling | PASS | 11.16s |  |
| artifacts | PASS | 1.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 603 --task-end 603
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
