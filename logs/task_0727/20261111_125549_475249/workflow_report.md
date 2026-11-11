# CUDA Agent Workflow Report

- Task ID: `727`
- Status: **SUCCESS**
- Operators: `["torch.arange", "nn.Upsample"]`
- Started: `2026-11-11T12:55:49`
- Finished: `2026-11-11T12:58:21`
- Duration: `152.03s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.89s |  |
| generation #1 | PASS | 36.81s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 40.16s | best; initial_candidate |
| profiling | PASS | 21.76s |  |
| artifacts | PASS | 42.29s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 727 --task-end 727
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
