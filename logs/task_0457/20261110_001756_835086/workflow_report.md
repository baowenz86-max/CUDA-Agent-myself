# CUDA Agent Workflow Report

- Task ID: `457`
- Status: **FAILED**
- Operators: `["torch.cat", "nn.LSTM"]`
- Started: `2026-11-10T00:17:56`
- Finished: `2026-11-10T00:21:17`
- Duration: `200.97s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `none`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 29.54s |  |
| generation #1 | PASS | 132.81s | Single Python → CUDA → C++ binding generation |
| initial candidate: environment candidate #1 | FAIL | 38.59s | rejected; terminal_failure; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-0co94qo0/probe.so']' timed out after 15 seconds
[TIME] Compilation took 18.70s
 |

## Failure Summary

Task terminated: environment; Compiling 3 files: binding.cpp, kernels/custom_ops.cu, kernels/custom_ops_binding.cpp
cudnn development dependency is not linkable: Command '['/usr/bin/g++-13', '-x', 'c++', '-', '-shared', '-Wl,--no-as-needed', '/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9', '-o', '/tmp/cuda-agent-cudnn-0co94qo0/probe.so']' timed out after 15 seconds
[TIME] Compilation took 18.70s


## Reproduction

```bash
python main.py --task-start 457 --task-end 457
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
