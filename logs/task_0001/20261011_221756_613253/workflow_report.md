# CUDA Agent Workflow Report

- Task ID: `1`
- Status: **FAILED**
- Operators: `["nn.LeakyReLU", "torch.pow", "torch.abs", "torch.einsum", "F.mish"]`
- Started: `2026-10-11T22:17:56`
- Finished: `2026-10-11T22:33:59`
- Duration: `963.32s`
- Full log: `workflow.log`

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| generation #1 | PASS | 29.41s | DeepSeek API |
| compile | PASS | 81.02s |  |
| compile after verification repair | PASS | 20.70s |  |
| compile after verification repair | PASS | 19.83s |  |
| verification | FAIL | 253.14s |  |
| generation #2 | PASS | 21.86s | DeepSeek API |
| compile | PASS | 104.26s |  |
| compile after verification repair | PASS | 43.50s |  |
| compile after verification repair | PASS | 19.74s |  |
| verification | FAIL | 206.62s |  |
| generation #3 | PASS | 19.34s | DeepSeek API |
| compile | PASS | 104.96s |  |
| compile after verification repair | PASS | 33.97s |  |
| compile after verification repair | PASS | 19.58s |  |
| verification | FAIL | 142.69s |  |

## Failure Summary

All generation attempts were exhausted. See workflow.log.

## Reproduction

```bash
python main.py --task-start 1 --task-end 1
```
