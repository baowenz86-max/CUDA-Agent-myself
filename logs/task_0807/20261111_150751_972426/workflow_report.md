# CUDA Agent Workflow Report

- Task ID: `807`
- Status: **SUCCESS**
- Operators: `["nn.ReLU", "torch.cos"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:12:48`
- Duration: `296.67s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 59.52s |  |
| generation #1 | PASS | 54.81s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 100.29s | best; initial_candidate |
| profiling | PASS | 28.58s |  |
| artifacts | PASS | 53.42s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 807 --task-end 807
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
