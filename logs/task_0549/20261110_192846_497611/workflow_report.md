# CUDA Agent Workflow Report

- Task ID: `549`
- Status: **SUCCESS**
- Operators: `["torch.max", "nn.PReLU"]`
- Started: `2026-11-10T19:28:46`
- Finished: `2026-11-10T19:31:35`
- Duration: `169.20s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.38s |  |
| generation #1 | PASS | 46.99s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 95.79s | best; initial_candidate |
| profiling | PASS | 12.37s |  |
| artifacts | PASS | 1.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 549 --task-end 549
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
