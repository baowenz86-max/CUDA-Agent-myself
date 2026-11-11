# CUDA Agent Workflow Report

- Task ID: `759`
- Status: **SUCCESS**
- Operators: `["F.softmax", "nn.Hardswish"]`
- Started: `2026-11-11T13:40:25`
- Finished: `2026-11-11T13:43:48`
- Duration: `203.61s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 40.27s |  |
| generation #1 | PASS | 67.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 66.23s | best; initial_candidate |
| profiling | PASS | 28.39s |  |
| artifacts | PASS | 1.11s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 759 --task-end 759
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
