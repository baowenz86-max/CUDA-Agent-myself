# CUDA Agent Workflow Report

- Task ID: `576`
- Status: **SUCCESS**
- Operators: `["torch.abs", "nn.MaxPool2d"]`
- Started: `2026-11-10T20:11:38`
- Finished: `2026-11-10T20:21:40`
- Duration: `602.01s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0004`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 31.37s |  |
| generation #1 | PASS | 52.93s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 32.00s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0576/kernels/custom_ops.cu(43): error: identifier "FLT_MAX" is undefined
      float max_val = -FLT_MAX;
                       ^
 |
| repair API call #1 | PASS | 11.69s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 21.89s | best; pipeline_progress; TypeError: custom_max_pool2d(): incompatible function arguments. The following argument types are supported:
    1. (x: torch.Tensor, kernel_size: typing.SupportsInt \| typing.SupportsIndex, stride: typing.SupportsInt \| typing.SupportsIndex, padding: typing.SupportsInt \| typing.SupportsIndex, dilation: typing.SupportsInt \| typing.SupportsIndex, ceil_mode: bool) -> torch.Tensor

Invoked with: tensor([[[[5.3513e-01, 1.4257e-01, 1.6688e+00,  ..., 9.3317e-01, |
| repair API call #2 | FAIL | 143.80s | Code repairs used: 1/3; base candidate #2; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #3 | PASS | 6.50s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | FAIL | 17.22s | best; pipeline_progress; AssertionError: The values for attribute 'shape' do not match: torch.Size([16, 3, 2050, 1026]) != torch.Size([16, 3, 2048, 1024]).

The above exception was the direct cause of the following exception: |
| repair API call #4 | FAIL | 120.58s | Code repairs used: 2/3; base candidate #3; API failed; best candidate retained; see api_calls.jsonl |
| repair API call #5 | PASS | 109.02s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | PASS | 27.06s | best; correctness_success |
| profiling | PASS | 20.49s |  |
| artifacts | PASS | 5.74s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 576 --task-end 576
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
