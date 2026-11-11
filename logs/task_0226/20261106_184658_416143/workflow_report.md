# CUDA Agent Workflow Report

- Task ID: `226`
- Status: **SUCCESS**
- Operators: `["torch.bmm", "nn.BatchNorm1d"]`
- Started: `2026-11-06T18:46:58`
- Finished: `2026-11-06T18:52:03`
- Duration: `305.49s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.51s |  |
| generation #1 | PASS | 53.28s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 148.27s | best |
| profiling | PASS | 31.71s |  |
| artifacts | PASS | 61.64s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 226 --task-end 226
```
