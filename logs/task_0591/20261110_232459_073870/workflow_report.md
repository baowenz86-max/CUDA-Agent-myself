# CUDA Agent Workflow Report

- Task ID: `591`
- Status: **SUCCESS**
- Operators: `["F.gelu", "nn.Linear"]`
- Started: `2026-11-10T23:24:59`
- Finished: `2026-11-10T23:31:25`
- Duration: `386.85s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 44.30s |  |
| generation #1 | PASS | 109.20s | Single Python → CUDA → C++ binding generation |
| initial candidate: sources candidate #1 | FAIL | 0.07s | best; initial_candidate; linear_forward: py::arg order/name/default differs |
| repair API call #1 | PASS | 17.15s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 131.49s | best; pipeline_progress; AssertionError: The values for attribute 'shape' do not match: torch.Size([512000, 256]) != torch.Size([512, 1000, 256]).

The above exception was the direct cause of the following exception: |
| repair API call #2 | PASS | 15.72s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 56.73s | best; correctness_success |
| profiling | PASS | 10.50s |  |
| artifacts | PASS | 1.06s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 591 --task-end 591
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
