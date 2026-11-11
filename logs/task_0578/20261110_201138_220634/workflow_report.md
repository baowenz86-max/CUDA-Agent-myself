# CUDA Agent Workflow Report

- Task ID: `578`
- Status: **FAILED**
- Operators: `["F.log_softmax", "torch.index_select"]`
- Started: `2026-11-10T20:11:38`
- Finished: `2026-11-10T20:12:06`
- Duration: `28.42s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 28.16s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0578/utils/preflight.py", line 110, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 3, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0578/20261110_201138_220634/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 578 --task-end 578
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
