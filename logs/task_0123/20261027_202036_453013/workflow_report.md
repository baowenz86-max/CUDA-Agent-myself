# CUDA Agent Workflow Report

- Task ID: `123`
- Status: **FAILED**
- Operators: `["nn.BCEWithLogitsLoss", "nn.LSTM", "torch.exp"]`
- Started: `2026-10-27T20:20:36`
- Finished: `2026-10-27T20:31:31`
- Duration: `655.24s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.48s |  |
| generation #1 | PASS | 289.27s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.03s | best; custom_ops_binding.cpp: prohibited torch computation in generated C++ |
| repair API call #1 | PASS | 49.12s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 13.59s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0123/kernels/custom_ops_binding.cpp:7:10: fatal error: cudnn.h: No such file or directory
    7 \| #include <cudnn.h>
      \|          ^~~~~~~~~
compilation terminated. |
| repair API call #2 | PASS | 175.87s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 39.52s | best; RuntimeError: lstm_forward kernel launch failed: invalid argument

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0123/20261027_202036_453013/correctness/verification.json |
| repair API call #3 | PASS | 71.66s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 12.52s | best; AssertionError: The values for attribute 'shape' do not match: torch.Size([16, 1]) != torch.Size([1024, 1]).

The above exception was the direct cause of the following exception: |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 4, stage verification: {"actual": {"type": "tensor", "shape": [16, 1], "dtype": "torch.float32", "device": "cuda:0", "stride": [1, 1], "numel": 16}, "expected": {"type": "tensor", "shape": [1024, 1], "dtype": "torch.float32", "device": "cuda:0", "stride": [1, 1], "numel": 1024}, "kind": "tensor_structure", "path": "output", "message": "The values for attribute 'shape' do not match: torch.Size([16, 1]) != torch.Size([1024, 1])."}

## Reproduction

```bash
python main.py --task-start 123 --task-end 123
```
