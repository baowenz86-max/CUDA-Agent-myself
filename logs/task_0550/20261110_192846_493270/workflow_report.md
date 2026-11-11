# CUDA Agent Workflow Report

- Task ID: `550`
- Status: **SUCCESS**
- Operators: `["torch.equal", "nn.MaxPool2d"]`
- Started: `2026-11-10T19:28:46`
- Finished: `2026-11-10T19:31:26`
- Duration: `159.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.64s |  |
| generation #1 | PASS | 50.70s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 50.48s | best; initial_candidate |
| profiling | PASS | 24.51s |  |
| artifacts | PASS | 25.24s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 550 --task-end 550
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
