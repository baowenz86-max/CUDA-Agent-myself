# CUDA Agent Workflow Report

- Task ID: `832`
- Status: **FAILED**
- Operators: `["torch.mul", "torch.sign"]`
- Started: `2026-11-11T17:58:22`
- Finished: `2026-11-11T17:58:49`
- Duration: `26.61s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 26.58s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0832/utils/preflight.py", line 110, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 2, in _reference_debug_forward
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0832/20261111_175822_390123/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 832 --task-end 832
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
