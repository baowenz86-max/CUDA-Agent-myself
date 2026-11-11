# CUDA Agent Workflow Report

- Task ID: `821`
- Status: **SUCCESS**
- Operators: `["torch.argsort", "nn.ReflectionPad2d"]`
- Started: `2026-11-11T17:49:13`
- Finished: `2026-11-11T17:52:08`
- Duration: `174.24s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.37s |  |
| generation #1 | PASS | 54.17s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 50.26s | best; initial_candidate |
| profiling | PASS | 33.37s |  |
| artifacts | PASS | 1.91s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 821 --task-end 821
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
