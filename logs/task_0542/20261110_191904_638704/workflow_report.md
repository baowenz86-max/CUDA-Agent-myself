# CUDA Agent Workflow Report

- Task ID: `542`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "torch.baddbmm"]`
- Started: `2026-11-10T19:19:04`
- Finished: `2026-11-10T19:25:41`
- Duration: `396.74s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 49.92s |  |
| generation #1 | PASS | 71.62s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 2.90s | best; initial_candidate; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "embedding_weight". 
	Unexpected key(s) in state_dict: "embedding.weight". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0542/20261110_191904_638704/correctness/interface_precompile.json |
| repair API call #1 | PASS | 8.33s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 166.60s | best; pipeline_progress; RuntimeError: indices must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0542/20261110_191904_638704/correctness/verification.json |
| repair API call #2 | PASS | 45.13s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 32.99s | best; correctness_success |
| profiling | PASS | 15.81s |  |
| artifacts | PASS | 2.52s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 542 --task-end 542
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
