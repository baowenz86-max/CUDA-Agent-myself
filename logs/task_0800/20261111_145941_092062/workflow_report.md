# CUDA Agent Workflow Report

- Task ID: `800`
- Status: **FAILED**
- Operators: `["torch.isin", "nn.MaxPool2d"]`
- Started: `2026-11-11T14:59:41`
- Finished: `2026-11-11T15:03:37`
- Duration: `236.52s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 8.48s |  |
| generation #1 | PASS | 108.91s | Single Python → CUDA → C++ binding generation |
| initial candidate: environment candidate #1 | FAIL | 119.10s | rejected; terminal_failure; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-sbv87h4p/probe.so']' timed out after 15 seconds
[TIME] Compilation took 18.11s
 |

## Failure Summary

Task terminated: environment; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-sbv87h4p/probe.so']' timed out after 15 seconds
[TIME] Compilation took 18.11s


## Reproduction

```bash
python main.py --task-start 800 --task-end 800
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
