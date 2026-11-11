# CUDA Agent Workflow Report

- Task ID: `416`
- Status: **SUCCESS**
- Operators: `["torch.tril", "torch.selu"]`
- Started: `2026-11-09T23:23:34`
- Finished: `2026-11-09T23:28:23`
- Duration: `288.76s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.04s |  |
| generation #1 | PASS | 108.97s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 131.44s | best; initial_candidate |
| profiling | PASS | 35.32s |  |
| artifacts | PASS | 1.80s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 416 --task-end 416
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
