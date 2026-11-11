# CUDA Agent Workflow Report

- Task ID: `384`
- Status: **SUCCESS**
- Operators: `["torch.norm"]`
- Started: `2026-11-09T21:35:20`
- Finished: `2026-11-09T21:43:35`
- Duration: `495.32s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 94.62s |  |
| generation #1 | PASS | 84.53s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 118.72s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0384/kernels/custom_ops_binding.cpp:114:12: error: expected unqualified-id before string constant
  114 \|     extern "C" void norm_forward_launcher_strides(
      \|            ^~~
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0384/kernels/custom_ops_binding.cpp:123:5: error: ‘norm_forward_launcher_strides’ was not declared in this scope; did you mean ‘norm_forward_launcher’?
  123 \|     norm_forward_launcher_strides(
      \|     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      \|     norm_forward_launcher |
| repair API call #1 | PASS | 28.62s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 137.75s | best; correctness_success |
| profiling | PASS | 28.88s |  |
| artifacts | PASS | 1.71s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 384 --task-end 384
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
