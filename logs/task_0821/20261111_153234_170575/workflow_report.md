# CUDA Agent Workflow Report

- Task ID: `821`
- Status: **SUCCESS**
- Operators: `["torch.argsort", "nn.ReflectionPad2d"]`
- Started: `2026-11-11T15:32:34`
- Finished: `2026-11-11T15:35:43`
- Duration: `189.41s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 49.06s |  |
| generation #1 | PASS | 71.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 33.85s | best; initial_candidate |
| profiling | PASS | 19.30s |  |
| artifacts | PASS | 15.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 821 --task-end 821
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
