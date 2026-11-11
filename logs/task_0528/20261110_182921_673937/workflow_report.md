# CUDA Agent Workflow Report

- Task ID: `528`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.log_softmax"]`
- Started: `2026-11-10T18:29:21`
- Finished: `2026-11-10T18:31:48`
- Duration: `146.34s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.32s |  |
| generation #1 | PASS | 75.26s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 48.91s | best; initial_candidate |
| profiling | PASS | 16.73s |  |
| artifacts | PASS | 1.93s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 528 --task-end 528
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
