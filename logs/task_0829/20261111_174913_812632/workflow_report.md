# CUDA Agent Workflow Report

- Task ID: `829`
- Status: **SUCCESS**
- Operators: `["nn.MultiheadAttention", "torch.matmul"]`
- Started: `2026-11-11T17:49:13`
- Finished: `2026-11-11T17:58:21`
- Duration: `547.21s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 36.97s |  |
| generation #1 | PASS | 182.88s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 54.59s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0829/kernels/custom_ops.cu(263): error: "b" has already been declared in the current scope
      int b = tmp / M;
          ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0829/kernels/custom_ops.cu(267): error: a value of type "int" cannot be used to initialize an entity of type "const float *"
      const float* b_col = b + b * b_b_stride + n * b_n_stride;
                           ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0829/kernels/custom_ops_binding.cpp:122:26: error: could not convert ‘pybind11::make_tuple(Args&& ...) [with return_value_policy policy = pybind11::return_value_policy::automatic_reference; Args = {at::Tensor&, at::Tensor&}](attn_weights)’ from ‘pybind11::typing::Tuple<at::Tensor&, at::Tensor&>’ to ‘at::Tensor’
  122 \|     return py::make_tuple(attn_output, attn_weights);
      \|            ~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
      \|                          \| |
| repair API call #1 | PASS | 83.80s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 29.90s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 33094373 / 33554432 (98.6%)
Greatest absolute difference: 12.560643196105957 at index (15, 861, 758) (up to 0.01 allowed)
Mismatched elements: 33091955 / 33554432 (98.6%)
Greatest absolute difference: 11.76909351348877 at index (12, 924, 626) (up to 0.01 allowed)
Mismatched elements: 33094417 / 33554432 (98.6%)
Greatest absolute difference: 11.984163284301758 at index (18, 664, 74) (up to 0.01 allowed)
Mismatched elements: 33094344 / 33554432 (98.6%)
Greatest absolute difference: 9.510472297668457 at index (16, 852, 589) (up to 0.01 allowed)
Mismatched elements: 33094518 / 33554432 (98.6%)
Greatest absolute difference: 8.551423072814941 at index (9, 715, 0) (up to 0.01 allowed) |
| repair API call #2 | FAIL | 37.94s | Code repairs used: 1/3; base candidate #2; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #3 | PASS | 79.32s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 25.25s | best; correctness_success |
| profiling | PASS | 14.99s |  |
| artifacts | PASS | 1.14s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 829 --task-end 829
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
