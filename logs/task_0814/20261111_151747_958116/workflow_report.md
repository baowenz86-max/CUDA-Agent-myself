# CUDA Agent Workflow Report

- Task ID: `814`
- Status: **SUCCESS**
- Operators: `["nn.Upsample", "nn.ConstantPad2d"]`
- Started: `2026-11-11T15:17:47`
- Finished: `2026-11-11T15:24:19`
- Duration: `391.43s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 3.22s |  |
| generation #1 | PASS | 181.68s | Single Python → CUDA → C++ binding generation |
| initial candidate: compile candidate #1 | FAIL | 74.54s | best; initial_candidate; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0814/build/forced_compile/cuda_extension.so: undefined symbol: upsample_launcher
[TIME] Compilation took 31.56s |
| repair API call #1 | PASS | 24.39s | Code repairs used: 1/3; base candidate #1 |
| repair #1: interface candidate #2 | FAIL | 13.40s | best; pipeline_progress; ValueError: constant_pad2d_forward: compiled signatures differ from operator_contract: [['x: torch.Tensor', 'padding: collections.abc.Sequence[typing.SupportsInt \| typing.SupportsIndex]', 'value: typing.SupportsFloat \| typing.SupportsIndex']]

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0814/20261111_151747_958116/correctness/interface.json |
| repair API call #2 | PASS | 30.38s | Code repairs used: 2/3; base candidate #2 |
| repair #2: verification candidate #3 | PASS | 38.43s | best; correctness_success |
| profiling | PASS | 24.19s |  |
| artifacts | PASS | 1.04s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 814 --task-end 814
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
