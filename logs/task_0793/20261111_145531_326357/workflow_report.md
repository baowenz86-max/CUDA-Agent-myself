# CUDA Agent Workflow Report

- Task ID: `793`
- Status: **SUCCESS**
- Operators: `["torch.isin", "nn.ZeroPad2d"]`
- Started: `2026-11-11T14:55:31`
- Finished: `2026-11-11T14:58:57`
- Duration: `206.52s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 7.59s |  |
| generation #1 | PASS | 54.32s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 112.82s | best; initial_candidate |
| profiling | PASS | 30.14s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 793 --task-end 793
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
