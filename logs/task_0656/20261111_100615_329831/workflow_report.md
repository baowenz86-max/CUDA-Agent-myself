# CUDA Agent Workflow Report

- Task ID: `656`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "F.layer_norm"]`
- Started: `2026-11-11T10:06:15`
- Finished: `2026-11-11T10:11:49`
- Duration: `334.64s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 59.85s |  |
| generation #1 | PASS | 46.49s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 195.10s | best; initial_candidate |
| profiling | PASS | 31.32s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 656 --task-end 656
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
