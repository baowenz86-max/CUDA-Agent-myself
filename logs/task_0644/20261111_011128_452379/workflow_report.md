# CUDA Agent Workflow Report

- Task ID: `644`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.floor"]`
- Started: `2026-11-11T01:11:28`
- Finished: `2026-11-11T01:18:50`
- Duration: `441.74s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 69.33s |  |
| generation #1 | PASS | 27.29s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 50.93s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0644/kernels/custom_ops_binding.cpp:57:10: error: no matching function for call to ‘pybind11::module_::def(const char [6], <unresolved overloaded function type>, const char [18], pybind11::arg)’
   57 \|     m.def("floor", &floor,
      \|     ~~~~~^~~~~~~~~~~~~~~~~
   58 \|           "Elementwise floor", |
| repair API call #1 | PASS | 8.77s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 272.54s | best; correctness_success |
| profiling | PASS | 11.81s |  |
| artifacts | PASS | 1.01s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 644 --task-end 644
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
