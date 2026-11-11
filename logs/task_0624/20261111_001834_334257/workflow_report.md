# CUDA Agent Workflow Report

- Task ID: `624`
- Status: **SUCCESS**
- Operators: `["torch.sum", "torch.exp"]`
- Started: `2026-11-11T00:18:34`
- Finished: `2026-11-11T00:22:45`
- Duration: `250.99s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.56s |  |
| generation #1 | PASS | 37.84s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 54.01s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0624/kernels/custom_ops_binding.cpp:96:10: error: no matching function for call to ‘pybind11::module_::def(const char [4], <unresolved overloaded function type>, const char [25], pybind11::arg)’
   96 \|     m.def("exp", &exp, "Element-wise exponential",
      \|     ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   97 \|           py::arg("x")); |
| repair API call #1 | PASS | 11.89s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 107.94s | best; correctness_success |
| profiling | PASS | 15.97s |  |
| artifacts | PASS | 1.69s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 624 --task-end 624
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
