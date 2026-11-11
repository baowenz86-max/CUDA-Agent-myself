# CUDA Agent Workflow Report

- Task ID: `660`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.ConstantPad1d"]`
- Started: `2026-11-11T10:06:15`
- Finished: `2026-11-11T10:11:13`
- Duration: `297.73s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 35.82s |  |
| generation #1 | PASS | 43.31s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 105.20s | best; initial_candidate |
| profiling | PASS | 110.62s |  |
| artifacts | PASS | 2.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 660 --task-end 660
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
