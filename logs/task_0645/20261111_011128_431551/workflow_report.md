# CUDA Agent Workflow Report

- Task ID: `645`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.Identity"]`
- Started: `2026-11-11T01:11:28`
- Finished: `2026-11-11T01:17:48`
- Duration: `380.32s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 50.62s |  |
| generation #1 | PASS | 28.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 212.40s | best; initial_candidate |
| profiling | PASS | 87.31s |  |
| artifacts | PASS | 1.61s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 645 --task-end 645
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
