# CUDA Agent Workflow Report

- Task ID: `352`
- Status: **SUCCESS**
- Operators: `["nn.Upsample", "nn.ReflectionPad2d"]`
- Started: `2026-11-09T14:15:52`
- Finished: `2026-11-09T14:20:01`
- Duration: `249.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 54.58s |  |
| generation #1 | PASS | 85.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 85.55s | best; initial_candidate |
| profiling | PASS | 22.85s |  |
| artifacts | PASS | 1.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 352 --task-end 352
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
