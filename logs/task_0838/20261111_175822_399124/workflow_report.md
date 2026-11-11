# CUDA Agent Workflow Report

- Task ID: `838`
- Status: **SUCCESS**
- Operators: `["torch.acos", "nn.ReflectionPad2d"]`
- Started: `2026-11-11T17:58:22`
- Finished: `2026-11-11T18:01:09`
- Duration: `167.02s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.88s |  |
| generation #1 | PASS | 43.02s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 56.35s | best; initial_candidate |
| profiling | PASS | 24.06s |  |
| artifacts | PASS | 13.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 838 --task-end 838
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
