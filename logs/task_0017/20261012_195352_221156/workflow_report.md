# CUDA Agent Workflow Report

- Task ID: `17`
- Status: **SUCCESS**
- Operators: `["F.conv_transpose1d", "torch.min", "torch.erf"]`
- Started: `2026-10-12T19:53:52`
- Finished: `2026-10-12T20:00:25`
- Duration: `392.85s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 25.62s |  |
| generation #1 | PASS | 44.52s | DeepSeek API |
| verification candidate #1 | FAIL | 95.84s | best |
| repair API | FAIL | 0.00s | No changes applied; best candidate retained |
| generation #2 | FAIL | 45.48s | DeepSeek API |
| generation #3 | PASS | 47.64s | DeepSeek API |
| verification candidate #3 | PASS | 23.93s | best |
| profiling | PASS | 17.08s |  |
| artifacts | PASS | 5.94s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 17 --task-end 17
```
