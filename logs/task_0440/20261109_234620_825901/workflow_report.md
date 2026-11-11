# CUDA Agent Workflow Report

- Task ID: `440`
- Status: **SUCCESS**
- Operators: `["nn.InstanceNorm2d", "torch.relu"]`
- Started: `2026-11-09T23:46:20`
- Finished: `2026-11-09T23:48:26`
- Duration: `125.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.65s |  |
| generation #1 | PASS | 45.56s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 44.06s | best; initial_candidate |
| profiling | PASS | 13.94s |  |
| artifacts | PASS | 1.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 440 --task-end 440
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
