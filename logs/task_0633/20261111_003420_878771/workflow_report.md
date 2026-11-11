# CUDA Agent Workflow Report

- Task ID: `633`
- Status: **SUCCESS**
- Operators: `["torch.cat", "torch.argmax"]`
- Started: `2026-11-11T00:34:20`
- Finished: `2026-11-11T00:37:22`
- Duration: `182.00s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.24s |  |
| generation #1 | PASS | 44.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 74.41s | best; initial_candidate |
| profiling | PASS | 25.30s |  |
| artifacts | PASS | 12.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 633 --task-end 633
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
