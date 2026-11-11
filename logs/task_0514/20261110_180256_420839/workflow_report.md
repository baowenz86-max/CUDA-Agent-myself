# CUDA Agent Workflow Report

- Task ID: `514`
- Status: **SUCCESS**
- Operators: `["nn.LayerNorm", "torch.mean"]`
- Started: `2026-11-10T18:02:56`
- Finished: `2026-11-10T18:06:45`
- Duration: `229.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 28.21s |  |
| generation #1 | PASS | 115.94s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 21.63s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "weight", "bias". 
	Unexpected key(s) in state_dict: "layer_norm.weight", "layer_norm.bias". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0514/20261110_180256_420839/correctness/interface_precompile.json |
| repair API call #1 | PASS | 9.53s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 40.45s | best; correctness_success |
| profiling | PASS | 11.77s |  |
| artifacts | PASS | 1.59s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 514 --task-end 514
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
