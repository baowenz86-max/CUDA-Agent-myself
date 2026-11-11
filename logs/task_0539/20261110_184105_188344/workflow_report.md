# CUDA Agent Workflow Report

- Task ID: `539`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "nn.ReLU"]`
- Started: `2026-11-10T18:41:05`
- Finished: `2026-11-10T18:42:59`
- Duration: `114.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.50s |  |
| generation #1 | PASS | 35.89s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 41.69s | best; initial_candidate |
| profiling | PASS | 25.99s |  |
| artifacts | PASS | 5.30s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 539 --task-end 539
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
