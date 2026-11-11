# CUDA Agent Workflow Report

- Task ID: `662`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.ZeroPad2d"]`
- Started: `2026-11-11T10:14:24`
- Finished: `2026-11-11T10:18:25`
- Duration: `241.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.92s |  |
| generation #1 | PASS | 51.22s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 107.21s | best; initial_candidate |
| profiling | PASS | 72.38s |  |
| artifacts | PASS | 1.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 662 --task-end 662
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
