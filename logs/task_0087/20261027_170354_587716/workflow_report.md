# CUDA Agent Workflow Report

- Task ID: `87`
- Status: **FAILED**
- Operators: `["nn.GroupNorm", "nn.LSTM", "torch.div"]`
- Started: `2026-10-27T17:03:54`
- Finished: `2026-10-27T17:12:55`
- Duration: `541.15s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 45.04s |  |
| generation #1 | PASS | 179.24s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 76.85s | best; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0087/kernels/custom_ops_binding.cpp:174:10: error: conflicting declaration ‘auto y’
  174 \|     auto y = torch::empty_like(x_contig);
      \|          ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0087/kernels/custom_ops_binding.cpp:151:10: note: previous declaration as ‘at::Tensor y’ |
| repair API call #1 | PASS | 24.56s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 124.52s | best; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0087/20261027_170354_587716/correctness/verification.json |
| repair API call #2 | PASS | 9.69s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 38.85s | rejected; RuntimeError: lstm_forward: x must be contiguous

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0087/20261027_170354_587716/correctness/verification.json |
| repair API call #3 | PASS | 7.21s | Code repairs used: 3/3; base candidate #2 |
| repair #3: verification candidate #4 | FAIL | 34.83s | rejected; RuntimeError: lstm_forward: x must be contiguous

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0087/20261027_170354_587716/correctness/verification.json |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 2, stage verification: {"kind": "runtime", "message": "NameError: name 'cuda_extension' is not defined"}

## Reproduction

```bash
python main.py --task-start 87 --task-end 87
```
