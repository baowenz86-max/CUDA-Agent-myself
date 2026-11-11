# CUDA Agent Workflow Report

- Task ID: `356`
- Status: **SUCCESS**
- Operators: `["nn.BatchNorm2d", "nn.GroupNorm"]`
- Started: `2026-11-09T14:51:58`
- Finished: `2026-11-09T14:55:08`
- Duration: `190.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 13.75s |  |
| generation #1 | PASS | 84.63s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 2.77s | best; initial_candidate; KeyError: 'buffer name can\'t contain "."'

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0356/20261109_145158_469015/correctness/interface_precompile.json |
| repair API call #1 | PASS | 14.13s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 34.16s | best; correctness_success |
| profiling | PASS | 38.79s |  |
| artifacts | PASS | 1.76s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 356 --task-end 356
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
