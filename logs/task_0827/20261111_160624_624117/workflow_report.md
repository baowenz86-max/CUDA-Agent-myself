# CUDA Agent Workflow Report

- Task ID: `827`
- Status: **FAILED**
- Operators: `["torch.squeeze", "nn.ConstantPad2d"]`
- Started: `2026-11-11T16:06:24`
- Finished: `2026-11-11T16:07:48`
- Duration: `84.03s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 83.98s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0827/utils/preflight.py", line 82, in main
    raise RuntimeError(dependency['reason'])
RuntimeError: cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-x8db_tfp/probe.so']' timed out after 15 seconds

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0827/20261111_160624_624117/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 827 --task-end 827
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
