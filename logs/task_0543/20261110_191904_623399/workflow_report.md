# CUDA Agent Workflow Report

- Task ID: `543`
- Status: **SUCCESS**
- Operators: `["F.gelu", "torch.softmax"]`
- Started: `2026-11-10T19:19:04`
- Finished: `2026-11-10T19:24:22`
- Duration: `317.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.86s |  |
| generation #1 | PASS | 91.01s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 103.85s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0543/kernels/custom_ops.cu(106): error: identifier "add_bias_kernel" is undefined
          add_bias_kernel<<<grid, block, 0, stream>>>(output, bias, M, N);
          ^
 |
| repair API call #1 | PASS | 34.01s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 17.98s | best; correctness_success |
| profiling | PASS | 23.47s |  |
| artifacts | PASS | 1.33s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 543 --task-end 543
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
