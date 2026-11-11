# CUDA Agent Workflow Report

- Task ID: `697`
- Status: **SUCCESS**
- Operators: `["torch.mean", "nn.LeakyReLU"]`
- Started: `2026-11-11T11:06:54`
- Finished: `2026-11-11T11:12:10`
- Duration: `316.51s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 52.62s |  |
| generation #1 | PASS | 44.90s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 165.91s | best; initial_candidate |
| profiling | PASS | 46.42s |  |
| artifacts | PASS | 6.62s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 697 --task-end 697
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
