# CUDA Agent Workflow Report

- Task ID: `812`
- Status: **SUCCESS**
- Operators: `["nn.Upsample", "torch.rot90"]`
- Started: `2026-11-11T15:17:48`
- Finished: `2026-11-11T15:22:24`
- Duration: `276.41s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 38.40s |  |
| generation #1 | PASS | 120.23s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 100.77s | best; initial_candidate |
| profiling | PASS | 14.96s |  |
| artifacts | PASS | 1.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 812 --task-end 812
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
