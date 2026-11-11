# CUDA Agent Workflow Report

- Task ID: `553`
- Status: **SUCCESS**
- Operators: `["nn.Embedding", "nn.AvgPool2d"]`
- Started: `2026-11-10T19:33:27`
- Finished: `2026-11-10T19:40:31`
- Duration: `423.93s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 16.69s |  |
| generation #1 | PASS | 68.35s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 146.85s | best; initial_candidate; RuntimeError: indices must be a CPU tensor

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0553/20261110_193327_427197/correctness/verification.json |
| repair API call #1 | PASS | 83.86s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 31.35s | best; pipeline_progress; AssertionError: The values for attribute 'shape' do not match: torch.Size([16, 256, 32, 128]) != torch.Size([16, 512, 32, 64]).

The above exception was the direct cause of the following exception: |
| repair API call #2 | PASS | 41.33s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 24.37s | best; correctness_success |
| profiling | PASS | 9.98s |  |
| artifacts | PASS | 1.02s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 553 --task-end 553
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
