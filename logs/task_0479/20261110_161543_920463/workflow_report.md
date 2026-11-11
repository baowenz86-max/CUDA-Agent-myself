# CUDA Agent Workflow Report

- Task ID: `479`
- Status: **SUCCESS**
- Operators: `["torch.clamp", "nn.MaxPool2d"]`
- Started: `2026-11-10T16:15:43`
- Finished: `2026-11-10T16:19:24`
- Duration: `220.38s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.61s |  |
| generation #1 | PASS | 86.54s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 93.33s | best; initial_candidate |
| profiling | PASS | 35.24s |  |
| artifacts | PASS | 1.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 479 --task-end 479
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
