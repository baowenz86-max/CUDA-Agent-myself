# CUDA Agent Workflow Report

- Task ID: `638`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.Embedding"]`
- Started: `2026-11-11T00:43:27`
- Finished: `2026-11-11T00:46:39`
- Duration: `191.89s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.92s |  |
| generation #1 | PASS | 47.43s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 103.79s | best; initial_candidate |
| profiling | PASS | 18.07s |  |
| artifacts | PASS | 1.55s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 638 --task-end 638
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
