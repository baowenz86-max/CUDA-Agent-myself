# CUDA Agent Workflow Report

- Task ID: `808`
- Status: **SUCCESS**
- Operators: `["torch.stack", "F.pad"]`
- Started: `2026-11-11T15:07:51`
- Finished: `2026-11-11T15:15:12`
- Duration: `441.05s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 20.16s |  |
| generation #1 | PASS | 69.51s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 23.47s | best; initial_candidate; ValueError: constant_pad: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'pad: collections.abc.Sequence[typing.SupportsInt \| typing.SupportsIndex]', 'value: typing.SupportsFloat \| typing.SupportsIndex'], ['x: Tensor', 'pad: List[int]', 'value: float']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0808/20261111_150751_935525/correctness/interface.json |
| repair API call #1 | PASS | 22.96s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 228.63s | best; pipeline_progress; AssertionError: Tensor-likes are not close!

Mismatched elements: 187076651 / 201916800 (92.7%)
Greatest absolute difference: 8.133773803710938 at index (7, 0, 2, 1682, 286) (up to 0.01 allowed)
Mismatched elements: 187076435 / 201916800 (92.7%)
Greatest absolute difference: 8.128162384033203 at index (11, 0, 2, 617, 521) (up to 0.01 allowed)
Mismatched elements: 187077627 / 201916800 (92.7%)
Greatest absolute difference: 7.914200782775879 at index (14, 1, 2, 1314, 319) (up to 0.01 allowed)
Mismatched elements: 187076205 / 201916800 (92.7%)
Greatest absolute difference: 8.022879600524902 at index (9, 1, 0, 282, 607) (up to 0.01 allowed)
Mismatched elements: 187077293 / 201916800 (92.7%)
Greatest absolute difference: 8.601729393005371 at index (12, 1, 0, 96, 422) (up to 0.01 allowed) |
| repair API call #2 | PASS | 26.94s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 37.22s | best; correctness_success |
| profiling | PASS | 10.69s |  |
| artifacts | PASS | 0.97s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 808 --task-end 808
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
