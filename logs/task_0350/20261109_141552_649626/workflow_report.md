# CUDA Agent Workflow Report

- Task ID: `350`
- Status: **FAILED**
- Operators: `["nn.ConvTranspose1d", "nn.Sequential"]`
- Started: `2026-11-09T14:15:52`
- Finished: `2026-11-09T14:16:39`
- Duration: `47.17s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 46.94s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0350/utils/preflight.py", line 82, in main
    raise RuntimeError(dependency['reason'])
RuntimeError: cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-kk1brcym/probe.so']' timed out after 15 seconds

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0350/20261109_141552_649626/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 350 --task-end 350
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
