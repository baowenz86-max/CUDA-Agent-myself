# CUDA Agent Workflow Report

- Task ID: `15`
- Status: **SUCCESS**
- Operators: `["nn.PReLU", "torch.argmax", "F.interpolate"]`
- Started: `2026-10-12T18:50:51`
- Finished: `2026-10-12T18:54:16`
- Duration: `205.33s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 12.20s |  |
| generation #1 | PASS | 49.53s | DeepSeek API |
| interface candidate #1 | FAIL | 71.56s | best |
| verification candidate #2 | PASS | 30.20s | best |
| profiling | PASS | 29.00s |  |
| artifacts | PASS | 3.09s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 15 --task-end 15
```
