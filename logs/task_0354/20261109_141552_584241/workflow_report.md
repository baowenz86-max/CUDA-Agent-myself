# CUDA Agent Workflow Report

- Task ID: `354`
- Status: **SUCCESS**
- Operators: `["nn.ReplicationPad2d"]`
- Started: `2026-11-09T14:15:52`
- Finished: `2026-11-09T14:19:51`
- Duration: `239.26s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 52.06s |  |
| generation #1 | PASS | 55.76s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 111.92s | best; initial_candidate |
| profiling | PASS | 17.54s |  |
| artifacts | PASS | 1.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 354 --task-end 354
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
