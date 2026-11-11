# CUDA Agent Workflow Report

- Task ID: `478`
- Status: **SUCCESS**
- Operators: `["torch.softmax", "nn.Mish"]`
- Started: `2026-11-10T16:15:43`
- Finished: `2026-11-10T16:20:05`
- Duration: `261.66s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 17.44s |  |
| generation #1 | PASS | 99.40s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 76.78s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0478/kernels/custom_ops.cu(19): error: identifier "CUDART_INF_F" is undefined
      float thread_max = -CUDART_INF_F;
                          ^
 |
| repair API call #1 | PASS | 40.65s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 14.01s | best; correctness_success |
| profiling | PASS | 11.10s |  |
| artifacts | PASS | 2.18s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 478 --task-end 478
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
