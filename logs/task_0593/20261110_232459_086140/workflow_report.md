# CUDA Agent Workflow Report

- Task ID: `593`
- Status: **SUCCESS**
- Operators: `["torch.abs", "F.layer_norm"]`
- Started: `2026-11-10T23:24:59`
- Finished: `2026-11-10T23:31:07`
- Duration: `368.67s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 41.01s |  |
| generation #1 | PASS | 68.70s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 83.42s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0593/kernels/custom_ops.cu(219): error: more than one instance of overloaded function "abs_forward_launcher" has "C" linkage
  extern "C" void abs_forward_launcher(
                  ^
 |
| repair API call #1 | PASS | 17.32s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 132.95s | best; correctness_success |
| profiling | PASS | 18.94s |  |
| artifacts | PASS | 6.01s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 593 --task-end 593
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
