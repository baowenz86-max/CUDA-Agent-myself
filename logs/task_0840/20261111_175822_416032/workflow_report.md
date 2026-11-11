# CUDA Agent Workflow Report

- Task ID: `840`
- Status: **FAILED**
- Operators: `["nn.ConvTranspose1d", "torch.detach"]`
- Started: `2026-11-11T17:58:22`
- Finished: `2026-11-11T17:59:32`
- Duration: `69.97s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 69.93s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0840/utils/preflight.py", line 82, in main
    raise RuntimeError(dependency['reason'])
RuntimeError: cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-5vp0dlqu/probe.so']' timed out after 15 seconds

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0840/20261111_175822_416032/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 840 --task-end 840
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
