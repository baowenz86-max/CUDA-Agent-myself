# CUDA Agent Workflow Report

- Task ID: `465`
- Status: **FAILED**
- Operators: `["torch.sub", "torch.masked_fill"]`
- Started: `2026-11-10T15:30:31`
- Finished: `2026-11-10T15:30:37`
- Duration: `5.94s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 5.91s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0465/utils/preflight.py", line 110, in main
    for (checkpoint, value), stage in zip(tracer(*device_inputs(inputs, args.device)), stages):
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<reference-checkpoints>", line 3, in _reference_debug_forward
RuntimeError: expected self and mask to be on the same device, but got mask on cuda:0 and self on cpu

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0465/20261110_153031_833957/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 465 --task-end 465
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
