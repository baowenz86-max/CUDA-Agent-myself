# CUDA Agent Workflow Report

- Task ID: `784`
- Status: **SUCCESS**
- Operators: `["torch.matmul", "nn.GroupNorm"]`
- Started: `2026-11-11T14:38:45`
- Finished: `2026-11-11T14:44:19`
- Duration: `334.06s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.92s |  |
| generation #1 | PASS | 98.12s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 181.12s | best; initial_candidate |
| profiling | PASS | 27.63s |  |
| artifacts | PASS | 1.11s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 784 --task-end 784
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
