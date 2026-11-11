# CUDA Agent Workflow Report

- Task ID: `585`
- Status: **FAILED**
- Operators: `["F.layer_norm", "nn.ConvTranspose1d", "torch.unbind"]`
- Started: `2026-11-10T20:22:03`
- Finished: `2026-11-10T20:22:24`
- Duration: `20.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `0`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 20.73s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0585/utils/preflight.py", line 82, in main
    raise RuntimeError(dependency['reason'])
RuntimeError: cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-1xydwb84/probe.so']' timed out after 15 seconds

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0585/20261110_202203_331973/correctness/task_contract.json


## Reproduction

```bash
python main.py --task-start 585 --task-end 585
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
