# CUDA Agent Workflow Report

- Task ID: `351`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.scalar_tensor"]`
- Started: `2026-11-09T14:15:52`
- Finished: `2026-11-09T14:19:14`
- Duration: `201.64s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.72s |  |
| generation #1 | PASS | 71.49s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 54.58s | best; initial_candidate |
| profiling | PASS | 43.32s |  |
| artifacts | PASS | 2.24s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 351 --task-end 351
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
