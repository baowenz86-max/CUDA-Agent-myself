# CUDA Agent Workflow Report

- Task ID: `307`
- Status: **FAILED**
- Operators: `["F.gelu", "nn.PixelShuffle", "nn.Upsample"]`
- Started: `2026-11-07T14:55:22`
- Finished: `2026-11-07T15:14:19`
- Duration: `1137.11s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `0/3` (initial evaluation excluded)
- Candidates evaluated: `1`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 5.99s |  |
| generation #1 | PASS | 57.68s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 691.38s | best; NameError: name 'cuda_extension' is not defined

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0307/20261107_145522_500458/correctness/verification.json |
| repair API call #1 | FAIL | 159.20s | Code repairs used: 0/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #2 | FAIL | 221.56s | Code repairs used: 0/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #3 | FAIL | 0.46s | Code repairs used: 0/3; base candidate #1; API failed; best candidate retained; see api_calls.jsonl |

## Failure Summary

Repair API failed 3 consecutive times; code repairs used 0/3. Best candidate 1, stage verification: {"kind": "runtime", "message": "NameError: name 'cuda_extension' is not defined"}

## Reproduction

```bash
python main.py --task-start 307 --task-end 307
```
