# CUDA Agent Workflow Report

- Task ID: `636`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool1d", "torch.tanh"]`
- Started: `2026-11-11T00:43:27`
- Finished: `2026-11-11T00:46:13`
- Duration: `165.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.73s |  |
| generation #1 | PASS | 35.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 42.64s | best; initial_candidate |
| profiling | PASS | 17.08s |  |
| artifacts | PASS | 46.50s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 636 --task-end 636
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
