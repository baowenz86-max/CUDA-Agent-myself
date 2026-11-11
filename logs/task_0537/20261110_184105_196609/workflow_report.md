# CUDA Agent Workflow Report

- Task ID: `537`
- Status: **FAILED**
- Operators: `["torch.where", "torch.round", "torch.bucketize"]`
- Started: `2026-11-10T18:41:05`
- Finished: `2026-11-10T18:41:30`
- Duration: `25.67s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 25.64s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0537/utils/preflight.py", line 110, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 4, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but got boundaries is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_Tensor_bucketize)

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0537/20261110_184105_196609/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 537 --task-end 537
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
