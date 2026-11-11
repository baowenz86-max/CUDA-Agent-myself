# CUDA Agent Workflow Report

- Task ID: `466`
- Status: **SUCCESS**
- Operators: `["torch.matmul", "nn.AvgPool3d"]`
- Started: `2026-11-10T16:01:31`
- Finished: `2026-11-10T16:06:27`
- Duration: `296.14s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.46s |  |
| generation #1 | PASS | 65.31s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 25.64s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0466/kernels/avg_pool3d_binding.cpp:55:11: error: ‘py’ has not been declared
   55 \|           py::arg("x"));
      \|           ^~
ninja: build stopped: subcommand failed. |
| repair API call #1 | PASS | 13.59s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 108.26s | best; correctness_success |
| profiling | PASS | 59.03s |  |
| artifacts | PASS | 1.78s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 466 --task-end 466
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
