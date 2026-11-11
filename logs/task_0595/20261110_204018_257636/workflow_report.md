# CUDA Agent Workflow Report

- Task ID: `595`
- Status: **FAILED**
- Operators: `["torch.equal", "nn.AdaptiveAvgPool1d"]`
- Started: `2026-11-10T20:40:18`
- Finished: `2026-11-10T20:47:03`
- Duration: `405.52s`
- Full log: `workflow.log`
- Generation: one three-stage generation (Python → CUDA → C++ binding)
- Code repairs used: `3/3` (initial evaluation excluded)
- Candidates evaluated: `4`
- Best correctness candidate: `correctness/candidates/0001`
- Correctness evidence: `correctness/` (contract, fixtures, history and candidate snapshots)

## Stage Summary

| Stage | Status | Duration | Details |
|---|---:|---:|---|
| reference contract | PASS | 21.71s |  |
| generation #1 | PASS | 38.93s | Single Python → CUDA → C++ binding generation |
| initial candidate: verification candidate #1 | FAIL | 72.33s | best; initial_candidate; TypeError: adaptive_avg_pool1d(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, output_size: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[-0.0552, -0.5738,  1.5453,  ...,  0.2161, -0.1817, -1.5353], |
| repair API call #1 | PASS | 112.20s | Code repairs used: 1/3; base candidate #1 |
| repair #1: verification candidate #2 | FAIL | 29.62s | rejected; tie_keep_previous; TypeError: adaptive_avg_pool1d(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, output_size: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[-0.0552, -0.5738,  1.5453,  ...,  0.2161, -0.1817, -1.5353], |
| repair API call #2 | PASS | 16.11s | Code repairs used: 2/3; base candidate #1 |
| repair #2: verification candidate #3 | FAIL | 29.45s | rejected; tie_keep_previous; TypeError: adaptive_avg_pool1d(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, output_size: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[-0.0552, -0.5738,  1.5453,  ...,  0.2161, -0.1817, -1.5353], |
| repair API call #3 | PASS | 31.13s | Code repairs used: 3/3; base candidate #1 |
| repair #3: verification candidate #4 | FAIL | 48.67s | rejected; tie_keep_previous; TypeError: adaptive_avg_pool1d(): incompatible function arguments. The following argument types are supported:
    1. (input: at::Tensor, output_size: typing.SupportsInt \| typing.SupportsIndex) -> at::Tensor

Invoked with: tensor([[[-0.0552, -0.5738,  1.5453,  ...,  0.2161, -0.1817, -1.5353], |

## Failure Summary

Repair budget exhausted (3/3). Best candidate 1, stage verification: {"kind": "runtime", "message": "TypeError: adaptive_avg_pool1d(): incompatible function arguments. The following argument types are supported:\n    1. (input: at::Tensor, output_size: typing.SupportsInt | typing.SupportsIndex) -> at::Tensor\n\nInvoked with: tensor([[[-0.0552, -0.5738,  1.5453,  ...,  0.2161, -0.1817, -1.5353],\n         [ 1.0881,  0.1681, -1.2906,  ...,  0.1057,  1.2249, -0.1104],\n         [ 1.0065, -0.0795, -0.3797,  ..., -1.0135, -1.1449, -0.1994],\n         ...,\n         [-1.1116, -0.9580,  0.5711,  ..., -2.2006,  0.6355,  1.4507],\n         [ 1.6884, -0.9380, -2.5496,  ...,  0.7358, -0.3587,  0.6176],\n         [-1.7973,  1.0057, -0.6492,  ..., -0.5110,  0.7316, -0.1622]],\n\n        [[-0.0429, -0.2903,  0.1699,  ..., -0.5295, -1.1262, -0.3259],\n         [ 0.3840,  0.4246, -0.1647,  ..., -0.4041, -1.1913,  1.2104],\n         [-0.5334, -0.7382,  0.9327,  ...,  0.9501,  0.6221,  1.6541],\n         ...,\n         [-1.8233, -0.1448, -0.1156,  ...,  0.1701,  0.3323, -0.4482],\n         [-0.3670, -0.2265,  1.2545,  ...,  0.5069, -1.8042, -0.3867],\n         [-1.3789, -0.1522, -0.0841,  ...,  0.6579,  0.7540, -0.3947]],\n\n        [[-0.1972, -0.1378,  0.2821,  ..., -1.4387, -0.2961,  0.7353],\n         [ 0.7303, -1.0411, -2.0222,  ...,  0.0455, -0.4243, -0.1419],\n         [ 0.4053,  0.5663,  0.2905,  ...,  2.1509, -0.5511, -0.3972],\n         ...,\n         [ 2.1217,  1.0799,  0.9910,  ..., -0.8754, -0.2430,  1.3113],\n         [ 0.5924,  1.0886,  0.0480,  ..., -0.1497, -1.0924,  0.1021],\n         [ 0.3769,  0.3710, -1.2378,  ..., -0.3172, -0.0680,  1.2130]],\n\n        ...,\n\n        [[ 1.0305,  0.1326, -0.6125,  ..., -0.1673, -0.1221, -0.6152],\n         [ 3.2900,  0.0458, -1.2267,  ..., -0.2602,  0.5128, -0.9515],\n         [ 0.7697, -1.1904, -0.7940,  ..., -0.4411,  1.9176,  2.2687],\n         ...,\n         [-0.6217,  0.2768,  0.6124,  ..., -0.2827,  1.3139, -0.3606],\n         [-1.2984,  0.9871, -0.0483,  ...,  0.1101,  1.1162, -0.9567],\n         [-1.4252,  0.6951,  0.7598,  ..., -0.3068,  0.2774,  0.3267]],\n\n        [[ 0.2616, -3.0757,  0.7803,  ..., -0.7575,  0.3305,  0.8786],\n         [ 0.8338,  0.0695, -1.2999,  ..., -0.4487, -0.4354, -1.2502],\n         [ 1.9150, -0.1685, -1.5920,  ..., -0.4380, -0.0516, -0.6412],\n         ...,\n         [-1.5538,  1.5611,  0.8562,  ...,  0.5306,  0.3412, -1.3136],\n         [-0.3516,  0.2209,  0.1529,  ...,  1.7204,  2.1692,  0.4479],\n         [-0.1979, -0.2962,  2.2936,  ..., -0.2790,  0.0507, -0.4381]],\n\n        [[ 2.0073,  0.7806,  0.3657,  ...,  0.0846,  0.2491, -2.3191],\n         [ 1.3682, -2.0202, -0.6632,  ...,  0.2916, -1.4231, -0.4724],\n         [ 0.4814,  1.2309, -2.4676,  ..., -0.0308,  0.3119, -0.3438],\n         ...,\n         [-0.7855, -0.9056,  0.6940,  ...,  0.7375,  1.4408,  0.0477],\n         [-1.4602,  1.1881, -1.1373,  ..., -0.5722,  0.0482, -1.6722],\n         [ 0.2316, -0.6681, -0.3725,  ..., -0.1316, -1.2895,  0.0210]]],\n       device='cuda:0'), 64"}

## Reproduction

```bash
python main.py --task-start 595 --task-end 595
```

## Machine-readable metrics

See `workflow_report.json`; unknown historical metrics are null.
