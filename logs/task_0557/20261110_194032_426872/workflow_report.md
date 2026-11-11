# CUDA Agent Workflow Report

- Task ID: `557`
- Status: **SUCCESS**
- Operators: `["nn.Mish"]`
- Started: `2026-11-10T19:40:32`
- Finished: `2026-11-10T19:42:57`
- Duration: `145.38s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.64s |  |
| generation #1 | PASS | 31.27s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 65.84s | best; initial_candidate |
| profiling | PASS | 18.05s |  |
| artifacts | PASS | 1.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 557 --task-end 557
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
