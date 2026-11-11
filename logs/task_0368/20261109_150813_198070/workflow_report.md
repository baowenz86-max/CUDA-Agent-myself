# CUDA Agent Workflow Report

- Task ID: `368`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "F.log_softmax"]`
- Started: `2026-11-09T15:08:13`
- Finished: `2026-11-09T15:15:08`
- Duration: `414.86s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 39.93s |  |
| generation #1 | PASS | 96.42s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 245.59s | best; initial_candidate |
| profiling | PASS | 24.97s |  |
| artifacts | PASS | 7.88s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 368 --task-end 368
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
