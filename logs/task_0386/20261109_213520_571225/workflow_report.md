# CUDA Agent Workflow Report

- Task ID: `386`
- Status: **SUCCESS**
- Operators: `["nn.AvgPool3d", "torch.min"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:42:48`
- Duration: `447.53s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 102.62s |  |
| generation #1 | PASS | 93.09s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 223.40s | best; initial_candidate |
| profiling | PASS | 26.39s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 386 --task-end 386
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
