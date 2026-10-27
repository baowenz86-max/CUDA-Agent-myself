# CUDA Agent Workflow Report

- Task ID: `107`
- Status: **FAILED**
- Operators: `["torch.cumprod", "torch.cumsum", "torch.matmul"]`
- Started: `2026-10-27T19:56:44`
- Finished: `2026-10-27T19:56:47`
- Duration: `2.87s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 2.82s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0107/utils/preflight.py", line 87, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 7, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_mm)

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0107/20261027_195644_774004/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 107 --task-end 107
```
