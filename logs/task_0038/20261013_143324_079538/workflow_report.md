# CUDA Agent Workflow Report

- Task ID: `38`
- Status: **FAILED**
- Operators: `["torch.index_select", "torch.round", "nn.MaxPool2d"]`
- Started: `2026-10-13T14:33:24`
- Finished: `2026-10-13T14:33:50`
- Duration: `26.42s`
- Full log: `workflow.log`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 26.28s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0038/utils/preflight.py", line 87, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 2, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0038/20261013_143324_079538/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 38 --task-end 38
```
