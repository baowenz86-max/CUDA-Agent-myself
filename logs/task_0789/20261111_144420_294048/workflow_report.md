# CUDA Agent Workflow Report

- Task ID: `789`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.scalar_tensor"]`
- Started: `2026-11-11T14:44:20`
- Finished: `2026-11-11T14:49:11`
- Duration: `290.80s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.04s |  |
| generation #1 | PASS | 53.04s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 5.39s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "scalar_tensor". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0789/20261111_144420_294048/correctness/interface_precompile.json |
| repair API call #1 | PASS | 18.18s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 126.45s | best; correctness_success |
| profiling | PASS | 67.98s |  |
| artifacts | PASS | 1.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 789 --task-end 789
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
