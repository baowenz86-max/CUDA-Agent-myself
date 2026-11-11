# CUDA Agent Workflow Report

- Task ID: `614`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.gt"]`
- Started: `2026-11-11T00:07:06`
- Finished: `2026-11-11T00:09:54`
- Duration: `168.04s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.24s |  |
| generation #1 | PASS | 36.71s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 60.57s | best; initial_candidate |
| profiling | PASS | 28.05s |  |
| artifacts | PASS | 14.39s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 614 --task-end 614
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
