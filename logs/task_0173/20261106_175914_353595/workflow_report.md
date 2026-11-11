# CUDA Agent Workflow Report

- Task ID: `173`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "torch.abs"]`
- Started: `2026-11-06T17:59:14`
- Finished: `2026-11-06T18:03:21`
- Duration: `247.24s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.58s |  |
| generation #1 | PASS | 45.60s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 87.53s | best |
| profiling | PASS | 38.56s |  |
| artifacts | PASS | 62.89s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 173 --task-end 173
```
