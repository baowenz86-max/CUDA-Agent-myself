# CUDA Agent Workflow Report

- Task ID: `533`
- Status: **SUCCESS**
- Operators: `["torch.matmul", "nn.Softmax"]`
- Started: `2026-11-10T18:33:10`
- Finished: `2026-11-10T18:36:28`
- Duration: `198.52s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.58s |  |
| generation #1 | PASS | 50.00s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 67.18s | best; initial_candidate |
| profiling | PASS | 40.47s |  |
| artifacts | PASS | 18.23s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 533 --task-end 533
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
