# CUDA Agent Workflow Report

- Task ID: `773`
- Status: **SUCCESS**
- Operators: `["torch.stack", "torch.all"]`
- Started: `2026-11-11T14:05:37`
- Finished: `2026-11-11T14:08:02`
- Duration: `145.62s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 4.26s |  |
| generation #1 | PASS | 51.88s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 21.42s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0773/kernels/custom_ops.cu(60): error: function "stack_dim0_launcher" has already been defined (previous definition at line 8)
  extern "C" void stack_dim0_launcher(const float* x, const float* y, float* output,
                  ^
 |
| repair API call #1 | PASS | 9.46s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 31.89s | best; correctness_success |
| profiling | PASS | 25.56s |  |
| artifacts | PASS | 1.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 773 --task-end 773
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
