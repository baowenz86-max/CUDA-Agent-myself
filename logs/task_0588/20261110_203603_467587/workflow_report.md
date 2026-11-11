# CUDA Agent Workflow Report

- Task ID: `588`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm1d"]`
- Started: `2026-11-10T20:36:03`
- Finished: `2026-11-10T20:40:17`
- Duration: `253.59s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.26s |  |
| generation #1 | PASS | 56.98s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 7.13s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "weight", "bias", "running_mean", "running_var", "num_batches_tracked". 
	Unexpected key(s) in state_dict: "batch_norm.weight", "batch_norm.bias", "batch_norm.running_mean", "batch_norm.running_var", "batch_norm.num_batches_tracked". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0588/20261110_203603_467587/correctness/interface_precompile.json |
| repair API call #1 | PASS | 10.57s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 149.77s | best; correctness_success |
| profiling | PASS | 22.44s |  |
| artifacts | PASS | 1.10s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 588 --task-end 588
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
