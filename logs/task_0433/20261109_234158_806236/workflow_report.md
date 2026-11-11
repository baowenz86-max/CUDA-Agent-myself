# CUDA Agent Workflow Report

- Task ID: `433`
- Status: **SUCCESS**
- Operators: `["nn.Unfold", "torch.reshape"]`
- Started: `2026-11-09T23:41:58`
- Finished: `2026-11-09T23:46:19`
- Duration: `260.83s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 14.46s |  |
| generation #1 | PASS | 51.59s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 99.97s | best; initial_candidate; AssertionError: The values for attribute 'shape' do not match: torch.Size([64, 128016, 45]) != torch.Size([64, 384048, 15]).

The above exception was the direct cause of the following exception: |
| repair API call #1 | PASS | 11.49s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 46.09s | rejected; missing_normal_observations; RuntimeError: Requested reshape dimensions do not match tensor size

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0433/20261109_234158_806236/correctness/verification.json |
| repair API call #2 | PASS | 10.17s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 13.98s | best; correctness_success |
| profiling | PASS | 11.65s |  |
| artifacts | PASS | 1.07s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 433 --task-end 433
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
