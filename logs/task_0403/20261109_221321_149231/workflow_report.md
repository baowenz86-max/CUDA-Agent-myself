# CUDA Agent Workflow Report

- Task ID: `403`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.rsqrt"]`
- Started: `2026-11-09T22:13:21`
- Finished: `2026-11-09T22:19:12`
- Duration: `350.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.11s |  |
| generation #1 | PASS | 91.81s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 25.67s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "embedding_weight". 
	Unexpected key(s) in state_dict: "embedding.weight". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0403/20261109_221321_149231/correctness/interface_precompile.json |
| repair API call #1 | PASS | 12.55s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 57.93s | rejected; tie_keep_previous; KeyError: 'parameter name can\'t contain "."'

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0403/20261109_221321_149231/correctness/interface_precompile.json |
| repair API call #2 | PASS | 34.72s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 88.29s | best; correctness_success |
| profiling | PASS | 10.70s |  |
| artifacts | PASS | 1.08s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 403 --task-end 403
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
