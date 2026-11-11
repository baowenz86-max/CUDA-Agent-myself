# CUDA Agent Workflow Report

- Task ID: `462`
- Status: **SUCCESS**
- Operators: `["nn.AvgPool3d", "F.kl_div"]`
- Started: `2026-11-10T15:30:31`
- Finished: `2026-11-10T15:36:20`
- Duration: `348.31s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.12s |  |
| generation #1 | PASS | 72.95s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 66.18s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0462/kernels/custom_ops.cu(73): error: too many arguments in function call
          batch, channels);
          ^
 |
| repair API call #1 | PASS | 25.93s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 158.36s | best; correctness_success |
| profiling | PASS | 12.65s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 462 --task-end 462
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
