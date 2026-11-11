# CUDA Agent Workflow Report

- Task ID: `457`
- Status: **FAILED**
- Operators: `["torch.cat", "nn.LSTM"]`
- Started: `2026-11-10T15:09:42`
- Finished: `2026-11-10T15:17:32`
- Duration: `469.92s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 69.17s |  |
| generation #1 | PASS | 300.38s | Single Python → CUDA → C++ binding generation |
| initial candidate: interface candidate #1 | FAIL | 5.22s | best; initial_candidate; KeyError: 'parameter name can\'t contain "."'

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0457/20261110_150942_762279/correctness/interface_precompile.json |
| repair API call #1 | PASS | 19.67s | Code repairs used: 1/3; base candidate #1 |
| repair #1: compile candidate #2 | FAIL | 28.33s | rejected; pipeline_regression; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(226): error: duplicate parameter name
                                         int num_layers, int batch, int hidden) {
                                                                        ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(230): error: operand types are incompatible ("int" and "const float *")
      if (b >= batch \|\| l >= num_layers \|\| j >= hidden) return;
                                             ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(232): error: expression must have arithmetic or unscoped enum type
      output[(size_t)l * (2 * batch) * hidden + (size_t)out_b * hidden + j] = hidden[(size_t)l * batch * hidden + (size_t)b * hidden + j];
                                       ^
                                                                ^
                                                                                                         ^
                                                                                                                              ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(233): error: expression must have arithmetic or unscoped enum type
      output[(size_t)l * (2 * batch) * hidden + (size_t)(batch + out_b) * hidden + j] = cell[(size_t)l * batch * hidden + (size_t)b * hidden + j];
                                                                          ^
                                                                                                                 ^
                                                                                                                                      ^ |
| repair API call #2 | PASS | 16.45s | Code repairs used: 2/3; base candidate #1 |
| repair #2: compile candidate #3 | FAIL | 5.49s | rejected; pipeline_regression; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(226): error: duplicate parameter name
                                         int num_layers, int batch, int hidden) {
                                                                        ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(230): error: operand types are incompatible ("int" and "const float *")
      if (b >= batch \|\| l >= num_layers \|\| j >= hidden) return;
                                             ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(232): error: expression must have arithmetic or unscoped enum type
      output[(size_t)l * (2 * batch) * hidden + (size_t)out_b * hidden + j] = hidden[(size_t)l * batch * hidden + (size_t)b * hidden + j];
                                       ^
                                                                ^
                                                                                                         ^
                                                                                                                              ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(233): error: expression must have arithmetic or unscoped enum type
      output[(size_t)l * (2 * batch) * hidden + (size_t)(batch + out_b) * hidden + j] = cell[(size_t)l * batch * hidden + (size_t)b * hidden + j];
                                                                          ^
                                                                                                                 ^
                                                                                                                                      ^ |
| repair API call #3 | PASS | 18.25s | Code repairs used: 3/3; base candidate #1 |
| repair #3: compile candidate #4 | FAIL | 6.54s | rejected; pipeline_regression; /home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(226): error: duplicate parameter name
                                         int num_layers, int batch, int hidden) {
                                                                        ^

/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(230): error: operand types are incompatible ("int" and "const float *")
      if (b >= batch \|\| l >= num_layers \|\| j >= hidden) return;
                                             ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(232): error: expression must have arithmetic or unscoped enum type
      output[(size_t)l * (2 * batch) * hidden + (size_t)out_b * hidden + j] = hidden[(size_t)l * batch * hidden + (size_t)b * hidden + j];
                                       ^
                                                                ^
                                                                                                         ^
                                                                                                                              ^
/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/kernels/custom_ops.cu(233): error: expression must have arithmetic or unscoped enum type
      output[(size_t)l * (2 * batch) * hidden + (size_t)(batch + out_b) * hidden + j] = cell[(size_t)l * batch * hidden + (size_t)b * hidden + j];
                                                                          ^
                                                                                                                 ^
                                                                                                                                      ^ |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage interface: Traceback (most recent call last):
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/utils/preflight.py", line 43, in main
    reference, generated, fixture = initialize_models(directory, args.seed, args.device)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/utils/verification.py", line 71, in initialize_models
    cuda_model = generated.ModelNew(*fixture['init_inputs']).eval().to(device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scc/pb24111609/projects/CUDA-Agent/.task_workdirs/task_0457/model_new.py", line 20, in __init__
    setattr(self, f'lstm.weight_ih_l{layer}', nn.Parameter(torch.empty(4 * hidden_size, input_dim)))
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1997, in __setattr__
    self.register_parameter(name, value)
  File "/home/scc/pb24111609/projects/CUDA-Agent/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 615, in register_parameter
    raise KeyError('parameter name can\'t contain "."')
KeyError: 'parameter name can\'t contain "."'

[FAILED] /home/scc/pb24111609/projects/CUDA-Agent/logs/task_0457/20261110_150942_762279/correctness/interface_precompile.json


## Reproduction

```bash
python main.py --task-start 457 --task-end 457
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
