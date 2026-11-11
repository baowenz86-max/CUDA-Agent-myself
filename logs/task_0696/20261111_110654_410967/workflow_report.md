# CUDA Agent Workflow Report

- Task ID: `696`
- Status: **SUCCESS**
- Operators: `["torch.sum", "F.pad"]`
- Started: `2026-11-11T11:06:54`
- Finished: `2026-11-11T11:13:11`
- Duration: `376.73s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.86s |  |
| generation #1 | PASS | 59.69s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 83.43s | best; initial_candidate; RuntimeError: copy_to_cuda_forward: input must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0696/20261111_110654_410967/correctness/verification.json |
| repair API call #1 | PASS | 28.95s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 130.09s | best; correctness_success |
| profiling | PASS | 36.69s |  |
| artifacts | PASS | 1.08s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 696 --task-end 696
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
