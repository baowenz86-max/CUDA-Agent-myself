# CUDA Agent Workflow Report

- Task ID: `622`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.unsqueeze"]`
- Started: `2026-11-11T00:18:34`
- Finished: `2026-11-11T00:21:46`
- Duration: `192.07s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.72s |  |
| generation #1 | PASS | 35.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 107.21s | best; initial_candidate |
| profiling | PASS | 26.63s |  |
| artifacts | PASS | 13.94s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 622 --task-end 622
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
