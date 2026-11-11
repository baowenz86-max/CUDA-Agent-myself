# CUDA Agent Workflow Report

- Task ID: `766`
- Status: **SUCCESS**
- Operators: `["torch.max", "torch.normal"]`
- Started: `2026-11-11T13:50:45`
- Finished: `2026-11-11T13:53:46`
- Duration: `181.44s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.60s |  |
| generation #1 | PASS | 34.42s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 56.21s | best; initial_candidate |
| profiling | PASS | 25.15s |  |
| artifacts | PASS | 53.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 766 --task-end 766
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
