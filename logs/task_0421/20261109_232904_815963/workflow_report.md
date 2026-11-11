# CUDA Agent Workflow Report

- Task ID: `421`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.bitwise_xor"]`
- Started: `2026-11-09T23:29:04`
- Finished: `2026-11-09T23:32:23`
- Duration: `199.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.44s |  |
| generation #1 | PASS | 49.95s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 79.43s | best; initial_candidate |
| profiling | PASS | 20.68s |  |
| artifacts | PASS | 42.50s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 421 --task-end 421
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
