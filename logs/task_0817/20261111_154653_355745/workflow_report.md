# CUDA Agent Workflow Report

- Task ID: `817`
- Status: **SUCCESS**
- Operators: `["torch.cos", "nn.Identity"]`
- Started: `2026-11-11T15:46:53`
- Finished: `2026-11-11T15:51:48`
- Duration: `295.37s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 54.77s |  |
| generation #1 | PASS | 51.87s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 84.16s | best; initial_candidate |
| profiling | PASS | 35.52s |  |
| artifacts | PASS | 68.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 817 --task-end 817
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
