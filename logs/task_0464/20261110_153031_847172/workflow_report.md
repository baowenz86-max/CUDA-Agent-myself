# CUDA Agent Workflow Report

- Task ID: `464`
- Status: **SUCCESS**
- Operators: `["nn.LeakyReLU"]`
- Started: `2026-11-10T15:30:31`
- Finished: `2026-11-10T15:34:24`
- Duration: `232.95s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.69s |  |
| generation #1 | PASS | 53.63s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 73.64s | best; initial_candidate |
| profiling | PASS | 82.41s |  |
| artifacts | PASS | 2.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 464 --task-end 464
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
