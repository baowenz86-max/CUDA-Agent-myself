# CUDA Agent Workflow Report

- Task ID: `361`
- Status: **SUCCESS**
- Operators: `["nn.Hardswish"]`
- Started: `2026-11-09T15:08:13`
- Finished: `2026-11-09T15:15:56`
- Duration: `463.09s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 75.11s |  |
| generation #1 | PASS | 57.72s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 245.11s | best; initial_candidate; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0361/20261109_150813_228907/correctness/verification.json |
| repair API call #1 | PASS | 5.82s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 55.96s | best; correctness_success |
| profiling | PASS | 19.63s |  |
| artifacts | PASS | 1.70s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 361 --task-end 361
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
