# CUDA Agent Workflow Report

- Task ID: `745`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.addmm"]`
- Started: `2026-11-11T13:19:29`
- Finished: `2026-11-11T13:24:53`
- Duration: `324.37s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 32.08s |  |
| generation #1 | PASS | 178.40s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 24.90s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0745/kernels/custom_ops.cu(255): error: identifier "add_bias_strided_kernel" is undefined
      add_bias_strided_kernel<<<blocks2, threads2, 0, stream>>>(out, bias, M, N, bias_s0, bias_s1);
      ^
 |
| repair API call #1 | PASS | 18.35s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 55.90s | best; correctness_success |
| profiling | PASS | 12.64s |  |
| artifacts | PASS | 1.39s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 745 --task-end 745
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
