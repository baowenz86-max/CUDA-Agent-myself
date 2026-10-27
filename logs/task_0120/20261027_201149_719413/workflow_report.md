# CUDA Agent Workflow Report

- Task ID: `120`
- Status: **SUCCESS**
- Operators: `["nn.Unfold", "torch.sign"]`
- Started: `2026-10-27T20:11:49`
- Finished: `2026-10-27T20:15:56`
- Duration: `246.83s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 10.88s |  |
| generation #1 | PASS | 52.08s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | PASS | 159.37s | best |
| profiling | PASS | 22.91s |  |
| artifacts | PASS | 1.53s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 120 --task-end 120
```
