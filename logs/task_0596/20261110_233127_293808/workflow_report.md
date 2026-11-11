# CUDA Agent Workflow Report

- Task ID: `596`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.PixelShuffle"]`
- Started: `2026-11-10T23:31:27`
- Finished: `2026-11-10T23:35:47`
- Duration: `259.80s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 23.91s |  |
| generation #1 | PASS | 41.99s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 114.80s | best; initial_candidate |
| profiling | PASS | 71.16s |  |
| artifacts | PASS | 7.89s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 596 --task-end 596
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
