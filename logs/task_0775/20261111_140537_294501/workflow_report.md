# CUDA Agent Workflow Report

- Task ID: `775`
- Status: **SUCCESS**
- Operators: `["torch.stack", "nn.ModuleDict"]`
- Started: `2026-11-11T14:05:37`
- Finished: `2026-11-11T14:07:51`
- Duration: `133.97s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 30.48s |  |
| generation #1 | PASS | 44.80s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 30.92s | best; initial_candidate |
| profiling | PASS | 26.09s |  |
| artifacts | PASS | 1.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 775 --task-end 775
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
