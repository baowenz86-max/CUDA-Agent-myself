# CUDA Agent Workflow Report

- Task ID: `802`
- Status: **SUCCESS**
- Operators: `["F.softmax", "nn.PReLU"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:13:51`
- Duration: `359.24s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 51.82s |  |
| generation #1 | PASS | 62.09s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 52.91s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0802/kernels/custom_ops.cu(98): error: too many arguments in function call
          x, y, weight, N, C, H, W, stride_n, stride_c, stride_h, stride_w, stream);
                                 ^
 |
| repair API call #1 | PASS | 26.00s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 123.51s | best; correctness_success |
| profiling | PASS | 41.05s |  |
| artifacts | PASS | 1.77s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 802 --task-end 802
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
