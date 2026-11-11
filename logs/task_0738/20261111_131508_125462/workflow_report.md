# CUDA Agent Workflow Report

- Task ID: `738`
- Status: **SUCCESS**
- Operators: `["torch.stack", "torch.clone"]`
- Started: `2026-11-11T13:15:08`
- Finished: `2026-11-11T13:18:39`
- Duration: `211.79s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.11s |  |
| generation #1 | PASS | 34.18s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 43.14s | best; initial_candidate; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0738/20261111_131508_125462/correctness/verification.json |
| repair API call #1 | PASS | 5.44s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 89.74s | best; correctness_success |
| profiling | PASS | 15.75s |  |
| artifacts | PASS | 1.57s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 738 --task-end 738
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
