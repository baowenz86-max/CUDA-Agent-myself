# CUDA Agent Workflow Report

- Task ID: `512`
- Status: **SUCCESS**
- Operators: `["torch.prod", "nn.AvgPool1d"]`
- Started: `2026-11-10T18:02:56`
- Finished: `2026-11-10T18:05:55`
- Duration: `179.00s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.91s |  |
| generation #1 | PASS | 69.53s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 52.67s | best; initial_candidate |
| profiling | PASS | 17.38s |  |
| artifacts | PASS | 18.22s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 512 --task-end 512
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
