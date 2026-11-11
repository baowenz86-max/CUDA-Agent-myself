# CUDA Agent Workflow Report

- Task ID: `597`
- Status: **SUCCESS**
- Operators: `["nn.ChannelShuffle"]`
- Started: `2026-11-10T23:31:27`
- Finished: `2026-11-10T23:35:55`
- Duration: `268.22s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.14s |  |
| generation #1 | PASS | 34.59s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 104.86s | best; initial_candidate |
| profiling | PASS | 98.99s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 597 --task-end 597
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
