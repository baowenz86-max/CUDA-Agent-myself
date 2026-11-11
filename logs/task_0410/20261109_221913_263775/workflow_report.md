# CUDA Agent Workflow Report

- Task ID: `410`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.AdaptiveAvgPool2d"]`
- Started: `2026-11-09T22:19:13`
- Finished: `2026-11-09T22:22:30`
- Duration: `197.01s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.80s |  |
| generation #1 | PASS | 86.29s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 5.27s | best; initial_candidate; KeyError: 'parameter name can\'t contain "."'

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0410/20261109_221913_263775/correctness/interface_precompile.json |
| repair API call #1 | PASS | 13.20s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 2.78s | rejected; tie_keep_previous; RuntimeError: Error(s) in loading state_dict for ModelNew:
	Missing key(s) in state_dict: "embedding_weight". 

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0410/20261109_221913_263775/correctness/interface_precompile.json |
| repair API call #2 | PASS | 12.59s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 46.45s | best; correctness_success |
| profiling | PASS | 19.44s |  |
| artifacts | PASS | 4.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 410 --task-end 410
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
