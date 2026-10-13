# CUDA Agent Workflow Report

- Task ID: `48`
- Status: **SUCCESS**
- Operators: `["torch.einsum", "nn.AdaptiveAvgPool3d", "nn.LeakyReLU", "nn.Conv3d", "nn.ConvTranspose1d"]`
- Started: `2026-10-13T15:49:03`
- Finished: `2026-10-13T15:58:31`
- Duration: `568.06s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.21s |  |
| generation #1 | PASS | 145.22s | DeepSeek API |
| compile candidate #1 | FAIL | 12.03s | best |
| interface candidate #2 | FAIL | 37.51s | best |
| verification candidate #3 | PASS | 175.76s | best |
| profiling | PASS | 54.04s |  |
| artifacts | PASS | 1.67s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 48 --task-end 48
```
