# CUDA Agent Workflow Report

- Task ID: `489`
- Status: **FAILED**
- Operators: `["F.relu", "nn.AdaptiveAvgPool2d"]`
- Started: `2026-11-10T17:18:26`
- Finished: `2026-11-10T17:19:02`
- Duration: `36.88s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 36.85s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0489/utils/preflight.py", line 82, in main
    raise RuntimeError(dependency['reason'])
RuntimeError: cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-geszh5xe/probe.so']' timed out after 15 seconds

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0489/20261110_171826_066914/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 489 --task-end 489
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
