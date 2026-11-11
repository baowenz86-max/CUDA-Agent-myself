# CUDA Agent Workflow Report

- Task ID: `502`
- Status: **SUCCESS**
- Operators: `["torch.matmul", "torch.softmax"]`
- Started: `2026-11-10T17:45:45`
- Finished: `2026-11-10T17:49:24`
- Duration: `219.47s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 19.60s |  |
| generation #1 | PASS | 59.33s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 23.55s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0502/kernels/custom_ops.cu(86): error: identifier "CUDART_INF_F" is undefined
      float max_val = -CUDART_INF_F;
                       ^
 |
| repair API call #1 | PASS | 23.35s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 67.68s | best; correctness_success |
| profiling | PASS | 24.53s |  |
| artifacts | PASS | 1.35s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 502 --task-end 502
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
