# CUDA Agent Workflow Report

- Task ID: `551`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.div"]`
- Started: `2026-11-10T19:33:27`
- Finished: `2026-11-10T19:39:42`
- Duration: `375.36s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.68s |  |
| generation #1 | PASS | 65.11s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 141.57s | best; initial_candidate; RuntimeError: indices must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0551/20261110_193327_436243/correctness/verification.json |
| repair API call #1 | PASS | 27.60s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 50.18s | rejected; tie_keep_previous; RuntimeError: indices must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0551/20261110_193327_436243/correctness/verification.json |
| repair API call #2 | PASS | 22.19s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 27.61s | best; correctness_success |
| profiling | PASS | 10.29s |  |
| artifacts | PASS | 0.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 551 --task-end 551
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
