# CUDA Agent Workflow Report

- Task ID: `36`
- Status: **FAILED**
- Operators: `["torch.add", "torch.argmin", "torch.stft"]`
- Started: `2026-10-12T21:15:25`
- Finished: `2026-10-12T21:15:45`
- Duration: `20.38s`
- Full log: `workflow.log`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | FAIL | 20.27s |  |

## Failure Summary

Reference preflight failed: Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0036/utils/preflight.py", line 11, in <module>
    import torch
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/__init__.py", line 2694, in <module>
    from torch import (
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/nested/__init__.py", line 21, in <module>
    from ._internal.nested_tensor import _rebuild_njt, NestedTensor as _NestedTensor
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/nested/_internal/nested_tensor.py", line 8, in <module>
    from torch.nested._internal.nested_int import NestedIntNode
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/nested/_internal/nested_int.py", line 4, in <module>
    from torch.fx.experimental._constant_symnode import ConstantIntNode
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/fx/__init__.py", line 88, in <module>
    from torch.fx._symbolic_trace import (
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 28, in <module>
    from .proxy import ParameterProxy, Proxy, Scope, ScopeContextManager, TracerBase
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/fx/proxy.py", line 21, in <module>
    from torch._logging import getArtifactLogger
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/_logging/__init__.py", line 7, in <module>
    import torch._logging._registrations
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/_logging/_registrations.py", line 1, in <module>
    from ._internal import register_artifact, register_log
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/_logging/_internal.py", line 27, in <module>
    from torch._guards import CompileId
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/_guards.py", line 12, in <module>
    import unittest.mock
  File "/home/scc/pb24111609/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/unittest/mock.py", line 26, in <module>
    import asyncio
  File "/home/scc/pb24111609/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/asyncio/__init__.py", line 22, in <module>
    from .threads import *
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1322, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1262, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1532, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1506, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1639, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1594, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 842, in spec_from_file_location
  File "<frozen importlib._bootstrap>", line 599, in __init__
KeyboardInterrupt


## Reproduction

```bash
python main.py --task-start 36 --task-end 36
```
