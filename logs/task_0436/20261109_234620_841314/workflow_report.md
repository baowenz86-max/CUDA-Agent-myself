# CUDA Agent Workflow Report

- Task ID: `436`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.BatchNorm3d"]`
- Started: `2026-11-09T23:46:20`
- Finished: `2026-11-09T23:51:05`
- Duration: `285.15s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 44.02s |  |
| generation #1 | PASS | 68.97s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 14.60s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "embedding_weight", "bn_weight", "bn_bias", "bn_running_mean", "bn_running_var", "bn_num_batches_tracked". 
	Unexpected key(s) in state_dict: "embedding.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0436/20261109_234620_841314/correctness/interface_precompile.json |
| repair API call #1 | PASS | 12.67s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 41.04s | best; pipeline_progress; RuntimeError: expected scalar type Float but found Long

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0436/20261109_234620_841314/correctness/verification.json |
| repair API call #2 | PASS | 20.83s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 13.68s | rejected; tie_keep_previous; RuntimeError: expected scalar type Float but found Long

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0436/20261109_234620_841314/correctness/verification.json |
| repair API call #3 | PASS | 29.61s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | PASS | 25.36s | best; correctness_success |
| profiling | PASS | 12.77s |  |
| artifacts | PASS | 1.33s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 436 --task-end 436
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
