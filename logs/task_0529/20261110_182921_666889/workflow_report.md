# CUDA Agent Workflow Report

- Task ID: `529`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.amax"]`
- Started: `2026-11-10T18:29:21`
- Finished: `2026-11-10T18:32:18`
- Duration: `176.80s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.83s |  |
| generation #1 | PASS | 56.40s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 69.53s | best; initial_candidate |
| profiling | PASS | 16.78s |  |
| artifacts | PASS | 1.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 529 --task-end 529
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
