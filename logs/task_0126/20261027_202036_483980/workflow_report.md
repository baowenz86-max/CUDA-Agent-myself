# CUDA Agent Workflow Report

- Task ID: `126`
- Status: **SUCCESS**
- Operators: `["F.gelu", "nn.MaxPool1d"]`
- Started: `2026-10-27T20:20:36`
- Finished: `2026-10-27T20:26:48`
- Duration: `372.41s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 41.28s |  |
| generation #1 | PASS | 47.46s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 50.79s | best; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0126/20261027_202036_483980/correctness/verification.json |
| repair API call #1 | PASS | 5.11s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 182.62s | best |
| profiling | PASS | 29.29s |  |
| artifacts | PASS | 15.39s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 126 --task-end 126
```
