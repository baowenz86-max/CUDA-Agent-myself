# CUDA Agent Workflow Report

- Task ID: `40`
- Status: **SUCCESS**
- Operators: `["nn.ConvTranspose1d", "torch.roll", "F.linear"]`
- Started: `2026-10-13T14:45:19`
- Finished: `2026-10-13T15:02:03`
- Duration: `1003.80s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0007`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 24.88s |  |
| generation #1 | PASS | 121.74s | DeepSeek API |
| compile candidate #1 | FAIL | 57.26s | best |
| sources candidate #2 | FAIL | 0.03s | rejected |
| interface candidate #3 | FAIL | 54.41s | best |
| generation #2 | PASS | 93.05s | DeepSeek API |
| compile candidate #4 | FAIL | 38.45s | rejected |
| verification candidate #5 | FAIL | 50.61s | best |
| repair API | FAIL | 0.00s | No changes applied; best candidate retained |
| generation #3 | PASS | 160.64s | DeepSeek API |
| compile candidate #6 | FAIL | 25.74s | rejected |
| verification candidate #7 | PASS | 55.60s | best |
| profiling | PASS | 38.08s |  |
| artifacts | PASS | 2.15s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 40 --task-end 40
```
