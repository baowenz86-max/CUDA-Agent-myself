# CUDA Agent Workflow Report

- Task ID: `646`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.log_softmax"]`
- Started: `2026-11-11T01:20:27`
- Finished: `2026-11-11T01:24:38`
- Duration: `250.76s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.46s |  |
| generation #1 | PASS | 34.47s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 158.02s | best; initial_candidate |
| profiling | PASS | 27.88s |  |
| artifacts | PASS | 1.88s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 646 --task-end 646
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
