# CUDA Agent Workflow Report

- Task ID: `474`
- Status: **FAILED**
- Operators: `["nn.AdaptiveLogSoftmaxWithLoss", "torch.softmax", "torch.clone"]`
- Started: `2026-11-10T16:07:29`
- Finished: `2026-11-10T16:15:42`
- Duration: `493.61s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 2.92s |  |
| generation #1 | PASS | 139.40s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 161.79s | best; initial_candidate; TypeError: adaptive_log_softmax(): incompatible function arguments. The following argument types are supported:
    1. (x: torch.Tensor, head_weight: torch.Tensor, tail_weight0: torch.Tensor, tail_weight1: torch.Tensor, cutoffs: typing.SupportsInt \| typing.SupportsIndex, div_value: typing.SupportsFloat \| typing.SupportsIndex) -> torch.Tensor

Invoked with: tensor([[-1.1439,  0.4982, -0.5202,  ...,  0.0877, -0.6538,  0.8850], |
| repair API call #1 | PASS | 31.58s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 20.36s | rejected; missing_normal_observations; ValueError: adaptive_log_softmax: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'head_weight: torch.Tensor', 'tail_weight0: torch.Tensor', 'tail_weight1: torch.Tensor', 'cutoffs: collections.abc.Sequence[typing.SupportsInt \| typing.SupportsIndex]', 'div_value: typing.SupportsFloat \| typing.SupportsIndex']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0474/20261110_160729_079393/correctness/interface.json |
| repair API call #2 | PASS | 21.61s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 26.80s | best; pipeline_progress; AssertionError: The values for attribute 'shape' do not match: torch.Size([2048, 10001]) != torch.Size([2048, 10000]).

The above exception was the direct cause of the following exception: |
| repair API call #3 | PASS | 27.72s | Code repairs used: 3/3; base candidate #3 |
| repair #3: verification candidate #4 | FAIL | 61.00s | rejected; missing_normal_observations; RuntimeError: head_weight first dim must equal head_size

[FAIL] Structured report: /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0474/20261110_160729_079393/correctness/verification.json |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 3, stage verification: {"actual": {"type": "tensor", "shape": [2048, 10001], "dtype": "torch.float32", "device": "cuda:0", "stride": [10001, 1], "numel": 20482048}, "expected": {"type": "tensor", "shape": [2048, 10000], "dtype": "torch.float32", "device": "cuda:0", "stride": [10000, 1], "numel": 20480000}, "kind": "tensor_structure", "path": "output[0]", "message": "The values for attribute 'shape' do not match: torch.Size([2048, 10001]) != torch.Size([2048, 10000])."}

## Reproduction

```bash
python main.py --task-start 474 --task-end 474
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
