# CUDA Agent Workflow Report

- Task ID: `569`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.Flatten"]`
- Started: `2026-11-10T19:57:55`
- Finished: `2026-11-10T20:00:28`
- Duration: `153.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.20s |  |
| generation #1 | PASS | 42.42s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 72.13s | best; initial_candidate |
| profiling | PASS | 30.30s |  |
| artifacts | PASS | 2.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 569 --task-end 569
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
