# CUDA Agent Workflow Report

- Task ID: `376`
- Status: **SUCCESS**
- Operators: `["torch.selu", "torch.sigmoid"]`
- Started: `2026-11-09T20:55:23`
- Finished: `2026-11-09T21:04:42`
- Duration: `558.53s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.90s |  |
| generation #1 | PASS | 295.32s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 141.96s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0376/kernels/custom_ops.cu(413): error: function "linear_forward_launcher" has already been defined (previous definition at line 18)
  extern "C" void linear_forward_launcher(
                  ^
 |
| repair API call #1 | PASS | 18.98s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 54.90s | best; correctness_success |
| profiling | PASS | 14.88s |  |
| artifacts | PASS | 2.46s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 376 --task-end 376
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
