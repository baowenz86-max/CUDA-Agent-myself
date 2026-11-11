# CUDA Agent Workflow Report

- Task ID: `790`
- Status: **SUCCESS**
- Operators: `["torch.sum", "nn.PReLU"]`
- Started: `2026-11-11T14:44:20`
- Finished: `2026-11-11T14:50:49`
- Duration: `389.56s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.62s |  |
| generation #1 | PASS | 58.43s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 29.06s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0790/build/forced_compile/cuda_extension.so: undefined symbol: prelu_launcher
[TIME] Compilation took 24.23s |
| repair API call #1 | PASS | 13.61s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 165.76s | best; pipeline_progress; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0790/20261111_144420_303305/correctness/verification.json |
| repair API call #2 | PASS | 7.52s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 30.33s | best; correctness_success |
| profiling | PASS | 56.73s |  |
| artifacts | PASS | 1.26s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 790 --task-end 790
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
