# CUDA Agent Workflow Report

- Task ID: `655`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.Unfold"]`
- Started: `2026-11-11T01:27:54`
- Finished: `2026-11-11T01:32:20`
- Duration: `265.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.73s |  |
| generation #1 | PASS | 80.53s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 122.73s | best; initial_candidate |
| profiling | PASS | 45.73s |  |
| artifacts | PASS | 1.74s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 655 --task-end 655
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
