# CUDA Agent Workflow Report

- Task ID: `435`
- Status: **SUCCESS**
- Operators: `["nn.PixelUnshuffle"]`
- Started: `2026-11-09T23:41:58`
- Finished: `2026-11-09T23:45:56`
- Duration: `237.91s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 22.85s |  |
| generation #1 | PASS | 32.31s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 53.16s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0435/build/forced_compile/cuda_extension.so: undefined symbol: pixel_unshuffle_launcher
[TIME] Compilation took 28.55s |
| repair API call #1 | PASS | 11.50s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 100.75s | best; correctness_success |
| profiling | PASS | 15.66s |  |
| artifacts | PASS | 1.60s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 435 --task-end 435
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
