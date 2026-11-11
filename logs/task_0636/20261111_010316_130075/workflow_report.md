# CUDA Agent Workflow Report

- Task ID: `636`
- Status: **SUCCESS**
- Operators: `["nn.MaxPool1d", "torch.tanh"]`
- Started: `2026-11-11T01:03:16`
- Finished: `2026-11-11T01:09:31`
- Duration: `375.32s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 42.86s |  |
| generation #1 | PASS | 39.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 32.35s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0636/kernels/custom_ops_binding.cpp:106:10: error: no matching function for call to ‘pybind11::module_::def(const char [5], <unresolved overloaded function type>, const char [13], pybind11::arg)’
  106 \|     m.def("tanh", &tanh,
      \|     ~~~~~^~~~~~~~~~~~~~~
  107 \|           "Tanh forward", |
| repair API call #1 | PASS | 12.97s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 196.52s | best; correctness_success |
| profiling | PASS | 49.17s |  |
| artifacts | PASS | 1.65s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 636 --task-end 636
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
