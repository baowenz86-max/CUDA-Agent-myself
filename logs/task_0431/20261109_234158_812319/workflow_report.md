# CUDA Agent Workflow Report

- Task ID: `431`
- Status: **SUCCESS**
- Operators: `["nn.Softplus", "nn.ReflectionPad2d", "nn.SiLU"]`
- Started: `2026-11-09T23:41:58`
- Finished: `2026-11-09T23:45:19`
- Duration: `200.85s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.05s |  |
| generation #1 | PASS | 48.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 115.24s | best; initial_candidate |
| profiling | PASS | 17.35s |  |
| artifacts | PASS | 1.81s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 431 --task-end 431
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
