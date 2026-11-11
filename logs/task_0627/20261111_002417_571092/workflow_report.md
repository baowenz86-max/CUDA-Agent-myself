# CUDA Agent Workflow Report

- Task ID: `627`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.ZeroPad2d"]`
- Started: `2026-11-11T00:24:17`
- Finished: `2026-11-11T00:27:12`
- Duration: `174.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.96s |  |
| generation #1 | PASS | 50.42s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 86.09s | best; initial_candidate |
| profiling | PASS | 23.34s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 627 --task-end 627
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
