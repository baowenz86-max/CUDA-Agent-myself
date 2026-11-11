# CUDA Agent Workflow Report

- Task ID: `554`
- Status: **SUCCESS**
- Operators: `["torch.tril", "nn.MaxPool2d"]`
- Started: `2026-11-10T19:33:27`
- Finished: `2026-11-10T19:37:13`
- Duration: `225.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.49s |  |
| generation #1 | PASS | 57.70s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 141.66s | best; initial_candidate |
| profiling | PASS | 19.01s |  |
| artifacts | PASS | 1.68s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 554 --task-end 554
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
