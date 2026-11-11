# CUDA Agent Workflow Report

- Task ID: `522`
- Status: **SUCCESS**
- Operators: `["torch.equal", "nn.BatchNorm2d"]`
- Started: `2026-11-10T18:19:58`
- Finished: `2026-11-10T18:21:29`
- Duration: `90.65s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.71s |  |
| generation #1 | PASS | 47.74s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 28.37s | best; initial_candidate |
| profiling | PASS | 10.71s |  |
| artifacts | PASS | 1.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 522 --task-end 522
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
