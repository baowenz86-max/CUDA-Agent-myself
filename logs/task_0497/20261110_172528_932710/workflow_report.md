# CUDA Agent Workflow Report

- Task ID: `497`
- Status: **FAILED**
- Operators: `["torch.index_select", "torch.minimum"]`
- Started: `2026-11-10T17:25:28`
- Finished: `2026-11-10T17:25:32`
- Duration: `3.57s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 3.54s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0497/utils/preflight.py", line 110, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 2, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0497/20261110_172528_932710/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 497 --task-end 497
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
