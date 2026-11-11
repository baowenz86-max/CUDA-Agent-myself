# CUDA Agent Workflow Report

- Task ID: `565`
- Status: **SUCCESS**
- Operators: `["torch.abs", "torch.clamp"]`
- Started: `2026-11-10T19:49:10`
- Finished: `2026-11-10T19:54:04`
- Duration: `294.08s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 54.03s |  |
| generation #1 | PASS | 40.36s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 92.65s | best; initial_candidate |
| profiling | PASS | 104.52s |  |
| artifacts | PASS | 2.31s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 565 --task-end 565
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
