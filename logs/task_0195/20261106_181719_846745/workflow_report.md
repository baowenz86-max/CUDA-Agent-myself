# CUDA Agent Workflow Report

- Task ID: `195`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "torch.selu"]`
- Started: `2026-11-06T18:17:19`
- Finished: `2026-11-06T18:24:56`
- Duration: `456.48s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 49.47s |  |
| generation #1 | PASS | 60.75s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 138.51s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0195/kernels/custom_ops_binding.cpp:5:10: fatal error: cudnn.h: No such file or directory
    5 \| #include <cudnn.h>
      \|          ^~~~~~~~~
compilation terminated. |
| repair API call #1 | PASS | 50.27s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 82.45s | best |
| profiling | PASS | 73.25s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 195 --task-end 195
```
