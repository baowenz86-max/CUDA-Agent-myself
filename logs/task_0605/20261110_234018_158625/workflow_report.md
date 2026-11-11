# CUDA Agent Workflow Report

- Task ID: `605`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool1d", "torch.relu"]`
- Started: `2026-11-10T23:40:18`
- Finished: `2026-11-10T23:43:30`
- Duration: `192.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.63s |  |
| generation #1 | PASS | 54.15s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 94.98s | best; initial_candidate |
| profiling | PASS | 20.87s |  |
| artifacts | PASS | 19.78s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 605 --task-end 605
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
