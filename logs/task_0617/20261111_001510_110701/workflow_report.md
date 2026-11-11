# CUDA Agent Workflow Report

- Task ID: `617`
- Status: **SUCCESS**
- Operators: `["torch.equal", "nn.GLU"]`
- Started: `2026-11-11T00:15:10`
- Finished: `2026-11-11T00:17:37`
- Duration: `147.45s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.51s |  |
| generation #1 | PASS | 29.65s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 52.04s | best; initial_candidate |
| profiling | PASS | 59.67s |  |
| artifacts | PASS | 1.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 617 --task-end 617
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
