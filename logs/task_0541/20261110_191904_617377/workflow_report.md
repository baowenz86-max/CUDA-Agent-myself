# CUDA Agent Workflow Report

- Task ID: `541`
- Status: **SUCCESS**
- Operators: `["torch.bmm", "nn.GELU"]`
- Started: `2026-11-10T19:19:04`
- Finished: `2026-11-10T19:23:36`
- Duration: `271.94s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 62.01s |  |
| generation #1 | PASS | 60.27s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 76.22s | best; initial_candidate |
| profiling | PASS | 70.58s |  |
| artifacts | PASS | 2.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 541 --task-end 541
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
