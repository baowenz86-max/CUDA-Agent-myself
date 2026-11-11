# CUDA Agent Workflow Report

- Task ID: `652`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.LeakyReLU"]`
- Started: `2026-11-11T01:27:54`
- Finished: `2026-11-11T01:32:09`
- Duration: `254.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.78s |  |
| generation #1 | PASS | 47.66s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 132.05s | best; initial_candidate |
| profiling | PASS | 38.15s |  |
| artifacts | PASS | 1.72s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 652 --task-end 652
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
