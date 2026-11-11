# CUDA Agent Workflow Report

- Task ID: `398`
- Status: **SUCCESS**
- Operators: `["torch.softmax", "sub"]`
- Started: `2026-11-09T21:54:03`
- Finished: `2026-11-09T21:59:33`
- Duration: `329.81s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.17s |  |
| generation #1 | PASS | 69.32s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 183.53s | best; initial_candidate |
| profiling | PASS | 61.01s |  |
| artifacts | PASS | 1.73s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 398 --task-end 398
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
