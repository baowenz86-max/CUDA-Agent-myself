# CUDA Agent Workflow Report

- Task ID: `795`
- Status: **SUCCESS**
- Operators: `["torch.stack", "torch.relu"]`
- Started: `2026-11-11T14:55:31`
- Finished: `2026-11-11T14:59:13`
- Duration: `222.33s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.47s |  |
| generation #1 | PASS | 54.16s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 140.06s | best; initial_candidate |
| profiling | PASS | 12.05s |  |
| artifacts | PASS | 1.56s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 795 --task-end 795
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
