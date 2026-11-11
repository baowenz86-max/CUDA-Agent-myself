# CUDA Agent Workflow Report

- Task ID: `447`
- Status: **SUCCESS**
- Operators: `["nn.ConstantPad3d"]`
- Started: `2026-11-10T00:03:32`
- Finished: `2026-11-10T00:07:59`
- Duration: `267.16s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `1/3` (initial evaluation excluded)
- Candidates evaluated: `2`
- Best correctness candidate: `correctness/candidates/0002`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 37.04s |  |
| generation #1 | PASS | 66.95s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 76.26s | best; initial_candidate; ValueError: constant_pad3d: compiled signatures differ from operator_contract: [['input: torch.Tensor', 'pad: collections.abc.Sequence[typing.SupportsInt \| typing.SupportsIndex]']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0447/20261110_000332_331516/correctness/interface.json |
| repair API call #1 | PASS | 19.04s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | PASS | 40.03s | best; correctness_success |
| profiling | PASS | 23.65s |  |
| artifacts | PASS | 4.09s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 447 --task-end 447
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
