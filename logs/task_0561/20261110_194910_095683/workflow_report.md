# CUDA Agent Workflow Report

- Task ID: `561`
- Status: **SUCCESS**
- Operators: `["torch.isnan", "nn.PReLU"]`
- Started: `2026-11-10T19:49:10`
- Finished: `2026-11-10T19:54:59`
- Duration: `349.35s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 34.63s |  |
| generation #1 | PASS | 42.28s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 38.67s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0561/build/forced_compile/cuda_extension.so: undefined symbol: custom_prelu_launcher
[TIME] Compilation took 35.85s |
| repair API call #1 | PASS | 11.13s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 177.02s | best; correctness_success |
| profiling | PASS | 38.10s |  |
| artifacts | PASS | 7.25s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 561 --task-end 561
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
