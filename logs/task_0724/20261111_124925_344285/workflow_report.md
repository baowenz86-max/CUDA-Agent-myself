# CUDA Agent Workflow Report

- Task ID: `724`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.GELU"]`
- Started: `2026-11-11T12:49:25`
- Finished: `2026-11-11T12:52:02`
- Duration: `157.52s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.76s |  |
| generation #1 | PASS | 28.47s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 100.25s | best; initial_candidate |
| profiling | PASS | 15.38s |  |
| artifacts | PASS | 1.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 724 --task-end 724
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
