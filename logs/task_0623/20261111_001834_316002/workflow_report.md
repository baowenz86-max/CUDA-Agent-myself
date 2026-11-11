# CUDA Agent Workflow Report

- Task ID: `623`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.GELU"]`
- Started: `2026-11-11T00:18:34`
- Finished: `2026-11-11T00:21:44`
- Duration: `190.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.65s |  |
| generation #1 | PASS | 36.97s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 36.24s | best; initial_candidate |
| profiling | PASS | 23.51s |  |
| artifacts | PASS | 91.16s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 623 --task-end 623
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
