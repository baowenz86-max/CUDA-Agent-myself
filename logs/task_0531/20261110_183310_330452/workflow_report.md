# CUDA Agent Workflow Report

- Task ID: `531`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.clamp_min"]`
- Started: `2026-11-10T18:33:10`
- Finished: `2026-11-10T18:36:56`
- Duration: `226.35s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.66s |  |
| generation #1 | PASS | 98.39s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 100.69s | best; initial_candidate |
| profiling | PASS | 16.29s |  |
| artifacts | PASS | 2.27s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 531 --task-end 531
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
