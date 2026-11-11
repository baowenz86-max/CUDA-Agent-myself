# CUDA Agent Workflow Report

- Task ID: `721`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.bitwise_xor"]`
- Started: `2026-11-11T12:49:25`
- Finished: `2026-11-11T12:51:49`
- Duration: `144.04s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.13s |  |
| generation #1 | PASS | 36.92s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 41.44s | best; initial_candidate |
| profiling | PASS | 22.18s |  |
| artifacts | PASS | 40.33s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 721 --task-end 721
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
