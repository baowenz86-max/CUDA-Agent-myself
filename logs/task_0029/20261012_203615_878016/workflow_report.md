# CUDA Agent Workflow Report

- Task ID: `29`
- Status: **FAILED**
- Operators: `["torch.detach", "torch.unbind", "torch.max"]`
- Started: `2026-10-12T20:36:15`
- Finished: `2026-10-12T20:50:29`
- Duration: `853.85s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.85s |  |
| generation #1 | PASS | 118.83s | DeepSeek API |
| sources candidate #1 | FAIL | 0.03s | best |
| compile candidate #2 | FAIL | 147.28s | best |
| verification candidate #3 | FAIL | 69.86s | best |
| generation #2 | FAIL | 277.06s | DeepSeek API |
| generation #3 | FAIL | 138.46s | DeepSeek API |

## Failure Summary

Best candidate 3, stage verification: {"kind": "type", "path": "output[0]", "actual": {"type": "tuple", "items": [{"type": "tensor", "shape": [512, 128], "dtype": "torch.float32", "device": "cuda:0", "stride": [128, 1], "numel": 65536}, {"type": "tensor", "shape": [512, 128], "dtype": "torch.int64", "device": "cuda:0", "stride": [128, 1], "numel": 65536}]}, "expected": {"type": "max", "items": [{"type": "tensor", "shape": [512, 128], "dtype": "torch.float32", "device": "cuda:0", "stride": [128, 1], "numel": 65536}, {"type": "tensor", "shape": [512, 128], "dtype": "torch.int64", "device": "cuda:0", "stride": [128, 1], "numel": 65536}]}}

## Reproduction

```bash
python main.py --task-start 29 --task-end 29
```
