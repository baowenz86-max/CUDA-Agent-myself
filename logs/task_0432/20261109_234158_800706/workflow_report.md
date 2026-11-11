# CUDA Agent Workflow Report

- Task ID: `432`
- Status: **FAILED**
- Operators: `["torch.max", "F.one_hot"]`
- Started: `2026-11-09T23:41:58`
- Finished: `2026-11-09T23:45:03`
- Duration: `184.80s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 11.72s |  |
| generation #1 | PASS | 67.06s | Single Python → CUDA → C++ binding generation |
| initial candidate: environment candidate #1 | FAIL | 105.98s | rejected; terminal_failure; Compiling 5 files: binding.cpp, kernels/custom_ops.cu, kernels/max_dim1_indices_binding.cpp, kernels/max_pool2d_binding.cpp, kernels/one_hot_from_indices_binding.cpp
cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-jcpfulr_/probe.so']' timed out after 15 seconds
[TIME] Compilation took 18.33s
 |

## Failure Summary

Task terminated: environment; Compiling 5 files: binding.cpp, kernels/custom_ops.cu, kernels/max_dim1_indices_binding.cpp, kernels/max_pool2d_binding.cpp, kernels/one_hot_from_indices_binding.cpp
cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-jcpfulr_/probe.so']' timed out after 15 seconds
[TIME] Compilation took 18.33s


## Reproduction

```bash
python main.py --task-start 432 --task-end 432
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
