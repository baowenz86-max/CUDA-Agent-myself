# CUDA Agent Workflow Report

- Task ID: `716`
- Status: **SUCCESS**
- Operators: `["torch.mean", "torch.scatter"]`
- Started: `2026-11-11T12:41:31`
- Finished: `2026-11-11T12:48:21`
- Duration: `410.77s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `2/3` (initial evaluation excluded)
- Candidates evaluated: `3`
- Best correctness candidate: `correctness/candidates/0003`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 15.02s |  |
| generation #1 | PASS | 58.47s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 148.21s | best; initial_candidate; AssertionError: Tensor-likes are not close!

Mismatched elements: 6268999 / 268435456 (2.3%)
Greatest absolute difference: 7.284123420715332 at index (286, 28, 37, 37) (up to 0.01 allowed)
Mismatched elements: 6268665 / 268435456 (2.3%)
Greatest absolute difference: 8.241181373596191 at index (218, 9, 94, 32) (up to 0.01 allowed)
Mismatched elements: 6269081 / 268435456 (2.3%)
Greatest absolute difference: 7.541518211364746 at index (278, 7, 71, 28) (up to 0.01 allowed)
Mismatched elements: 6268910 / 268435456 (2.3%)
Greatest absolute difference: 7.759355545043945 at index (72, 5, 111, 10) (up to 0.01 allowed)
Mismatched elements: 6269002 / 268435456 (2.3%)
Greatest absolute difference: 7.160880088806152 at index (497, 15, 33, 121) (up to 0.01 allowed) |
| repair API call #1 | PASS | 34.85s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 6.94s | rejected; missing_normal_observations; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0716/kernels/custom_ops.cu(100): error: no instance of overloaded function "atomicMax" matches the argument list
            argument types are: (uint64_t *, uint64_t)
      atomicMax(&out_buffer[out_offset], packed);
      ^ |
| repair API call #2 | PASS | 45.06s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | PASS | 79.79s | best; correctness_success |
| profiling | PASS | 18.78s |  |
| artifacts | PASS | 1.12s | best effort |

## Failure Summary

None.

## Reproduction

```bash
python main.py --task-start 716 --task-end 716
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
