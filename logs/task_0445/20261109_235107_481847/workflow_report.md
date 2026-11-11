# CUDA Agent Workflow Report

- Task ID: `445`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.SyncBatchNorm"]`
- Started: `2026-11-09T23:51:07`
- Finished: `2026-11-09T23:55:33`
- Duration: `265.93s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 26.53s |  |
| generation #1 | PASS | 52.52s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 160.00s | best; initial_candidate |
| profiling | PASS | 25.13s |  |
| artifacts | PASS | 1.70s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 445 --task-end 445
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
