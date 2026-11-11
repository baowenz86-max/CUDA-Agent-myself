# CUDA Agent Workflow Report

- Task ID: `806`
- Status: **SUCCESS**
- Operators: `["torch.stack", "F.softmax"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:14:55`
- Duration: `423.63s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 55.03s |  |
| generation #1 | PASS | 79.76s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 226.06s | best; initial_candidate; RuntimeError: All input tensors must be on CPU

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0806/20261111_150751_967319/correctness/verification.json |
| repair API call #1 | PASS | 18.93s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 28.52s | best; correctness_success |
| profiling | PASS | 13.24s |  |
| artifacts | PASS | 1.63s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 806 --task-end 806
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
