# CUDA Agent Workflow Report

- Task ID: `399`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.gather"]`
- Started: `2026-11-09T21:54:03`
- Finished: `2026-11-09T21:59:24`
- Duration: `320.97s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.72s |  |
| generation #1 | PASS | 74.19s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 117.76s | best; initial_candidate |
| profiling | PASS | 52.05s |  |
| artifacts | PASS | 70.20s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 399 --task-end 399
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
