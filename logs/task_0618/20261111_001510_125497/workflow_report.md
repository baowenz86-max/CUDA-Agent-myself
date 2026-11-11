# CUDA Agent Workflow Report

- Task ID: `618`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool1d", "torch.softmax"]`
- Started: `2026-11-11T00:15:10`
- Finished: `2026-11-11T00:18:06`
- Duration: `176.07s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 15.56s |  |
| generation #1 | PASS | 37.29s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 103.40s | best; initial_candidate |
| profiling | PASS | 18.13s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 618 --task-end 618
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
