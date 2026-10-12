# CUDA Agent Workflow Report

- Task ID: `8`
- Status: **FAILED**
- Operators: `["torch.isinf", "nn.ConstantPad2d", "torch.rot90"]`
- Started: `2026-10-12T17:27:33`
- Finished: `2026-10-12T17:48:12`
- Duration: `1238.90s`
- Full log: `workflow.log`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 6.78s |  |
| generation #1 | PASS | 98.01s | DeepSeek API |
| compile candidate #1 | FAIL | 118.16s | best |
| verification candidate #2 | FAIL | 395.71s | best |
| verification candidate #3 | FAIL | 237.80s | best |
| generation #2 | FAIL | 176.67s | DeepSeek API |
| generation #3 | FAIL | 139.76s | DeepSeek API |

## Failure Summary

Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [512, 16, 128, 128], "dtype": "torch.float32", "device": "cuda:0", "stride": [262144, 16384, 128, 1], "numel": 134217728}, "expected": {"type": "tensor", "shape": [512, 16, 128, 126], "dtype": "torch.float32", "device": "cuda:0", "stride": [258048, 16128, 126, 1], "numel": 132120576}, "kind": "tensor_structure", "path": "output", "message": "The values for attribute 'shape' do not match: torch.Size([512, 16, 128, 128]) != torch.Size([512, 16, 128, 126])."}

## Reproduction

```bash
python main.py --task-start 8 --task-end 8
```
