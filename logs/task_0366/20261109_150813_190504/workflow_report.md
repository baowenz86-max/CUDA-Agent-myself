# CUDA Agent Workflow Report

- Task ID: `366`
- Status: **SUCCESS**
- Operators: `["torch.equal", "torch.sin"]`
- Started: `2026-11-09T15:08:13`
- Finished: `2026-11-09T15:14:26`
- Duration: `373.29s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.74s |  |
| generation #1 | PASS | 58.18s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 31.98s | best; initial_candidate; RuntimeError: custom_equal_cpu_int64: x must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0366/20261109_150813_190504/correctness/verification.json |
| repair API call #1 | PASS | 17.82s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 159.92s | best; correctness_success |
| profiling | PASS | 80.99s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 366 --task-end 366
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
