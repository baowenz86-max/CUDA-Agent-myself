# CUDA Agent Workflow Report

- Task ID: `698`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.ReLU"]`
- Started: `2026-11-11T11:06:54`
- Finished: `2026-11-11T11:09:53`
- Duration: `179.49s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.18s |  |
| generation #1 | PASS | 45.58s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 65.32s | best; initial_candidate |
| profiling | PASS | 38.28s |  |
| artifacts | PASS | 20.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 698 --task-end 698
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
