# CUDA Agent Workflow Report

- Task ID: `825`
- Status: **FAILED**
- Operators: `["torch.index_select", "nn.ConvTranspose1d"]`
- Started: `2026-11-11T15:32:34`
- Finished: `2026-11-11T15:33:59`
- Duration: `85.65s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 85.62s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0825/utils/preflight.py", line 82, in main
    raise RuntimeError(dependency['reason'])
RuntimeError: cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-a2h1jvra/probe.so']' timed out after 15 seconds

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0825/20261111_153234_192537/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 825 --task-end 825
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
