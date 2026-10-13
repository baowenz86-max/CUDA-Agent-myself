# CUDA Agent Workflow Report

- Task ID: `32`
- Status: **FAILED**
- Operators: `["torch.clip", "torch.bucketize", "torch.bmm"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T14:45:47`
- Duration: `27.87s`
- Full log: `workflow.log`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 27.84s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0032/utils/preflight.py", line 87, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 3, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but got boundaries is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_Tensor_bucketize)

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0032/20261013_144519_784905/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 32 --task-end 32
```
