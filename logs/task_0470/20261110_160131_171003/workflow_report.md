# CUDA Agent Workflow Report

- Task ID: `470`
- Status: **SUCCESS**
- Operators: `["nn.functional", "torch.abs"]`
- Started: `2026-11-10T16:01:31`
- Finished: `2026-11-10T16:07:17`
- Duration: `346.66s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.51s |  |
| generation #1 | PASS | 78.37s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 111.30s | best; initial_candidate; RuntimeError: Input must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0470/20261110_160131_171003/correctness/verification.json |
| repair API call #1 | PASS | 23.15s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 76.41s | best; correctness_success |
| profiling | PASS | 41.38s |  |
| artifacts | PASS | 1.42s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 470 --task-end 470
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
