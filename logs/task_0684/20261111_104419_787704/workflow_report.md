# CUDA Agent Workflow Report

- Task ID: `684`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.split"]`
- Started: `2026-11-11T10:44:19`
- Finished: `2026-11-11T10:46:15`
- Duration: `115.36s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.38s |  |
| generation #1 | PASS | 57.58s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 36.83s | best; initial_candidate |
| profiling | PASS | 15.89s |  |
| artifacts | PASS | 1.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 684 --task-end 684
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
