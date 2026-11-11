# CUDA Agent Workflow Report

- Task ID: `424`
- Status: **SUCCESS**
- Operators: `["nn.PReLU"]`
- Started: `2026-11-09T23:29:04`
- Finished: `2026-11-09T23:33:25`
- Duration: `261.08s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 18.46s |  |
| generation #1 | PASS | 35.16s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 7.36s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "prelu_weight". 
	Unexpected key(s) in state_dict: "prelu.weight". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0424/20261109_232904_825719/correctness/interface_precompile.json |
| repair API call #1 | PASS | 6.61s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 139.40s | best; correctness_success |
| profiling | PASS | 52.34s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 424 --task-end 424
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
