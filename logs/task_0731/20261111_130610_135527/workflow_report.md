# CUDA Agent Workflow Report

- Task ID: `731`
- Status: **SUCCESS**
- Operators: `["F.softmax", "torch.mul"]`
- Started: `2026-11-11T13:06:10`
- Finished: `2026-11-11T13:12:37`
- Duration: `387.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 43.56s |  |
| generation #1 | PASS | 50.11s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 229.52s | best; initial_candidate; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0731/20261111_130610_135527/correctness/verification.json |
| repair API call #1 | PASS | 5.36s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 46.30s | best; correctness_success |
| profiling | PASS | 11.50s |  |
| artifacts | PASS | 1.05s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 731 --task-end 731
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
