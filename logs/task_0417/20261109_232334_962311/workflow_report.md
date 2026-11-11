# CUDA Agent Workflow Report

- Task ID: `417`
- Status: **SUCCESS**
- Operators: `["F.max_pool2d_with_indices"]`
- Started: `2026-11-09T23:23:34`
- Finished: `2026-11-09T23:29:03`
- Duration: `328.65s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.91s |  |
| generation #1 | PASS | 54.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 92.58s | best; initial_candidate; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0417/20261109_232334_962311/correctness/verification.json |
| repair API call #1 | PASS | 24.94s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 72.15s | best; correctness_success |
| profiling | PASS | 54.56s |  |
| artifacts | PASS | 1.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 417 --task-end 417
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
