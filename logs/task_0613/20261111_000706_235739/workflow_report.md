# CUDA Agent Workflow Report

- Task ID: `613`
- Status: **SUCCESS**
- Operators: `["nn.InstanceNorm1d"]`
- Started: `2026-11-11T00:07:06`
- Finished: `2026-11-11T00:10:39`
- Duration: `213.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.56s |  |
| generation #1 | PASS | 40.71s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 108.80s | best; initial_candidate |
| profiling | PASS | 30.42s |  |
| artifacts | PASS | 1.85s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 613 --task-end 613
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
