# CUDA Agent Workflow Report

- Task ID: `816`
- Status: **SUCCESS**
- Operators: `["nn.Upsample", "torch.mean"]`
- Started: `2026-11-11T15:46:53`
- Finished: `2026-11-11T15:51:50`
- Duration: `296.98s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 43.11s |  |
| generation #1 | PASS | 78.84s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 130.32s | best; initial_candidate |
| profiling | PASS | 31.23s |  |
| artifacts | PASS | 13.39s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 816 --task-end 816
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
