import argparse
import gc

import torch

from model import Model
from model_new import ModelNew


CASES = [
    # batch, channels, height, width, groups
    (1, 4, 32, 32, 1),
    (2, 16, 64, 64, 2),
    (4, 32, 128, 64, 4),
    (1, 64, 64, 128, 8),
    (8, 32, 32, 32, 4),
]


def compare(x, groups, description):
    channels = x.shape[1]

    baseline = Model(channels, groups, channels).eval().cuda()
    extension = ModelNew(channels, groups, channels).eval().cuda()
    extension.load_state_dict(baseline.state_dict())

    with torch.no_grad():
        expected = baseline(x)
        actual = extension(x)
        torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-2,
        rtol=1e-2,
    )

    print(f"[PASS] {description}: shape={tuple(x.shape)}, groups={groups}")

    del baseline, extension, expected, actual, x
    gc.collect()
    torch.cuda.empty_cache()


def random_tests(num_seeds):
    for seed in range(num_seeds):
        batch, channels, height, width, groups = CASES[seed % len(CASES)]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        x = torch.randn(
            batch,
            channels,
            height,
            width,
            device="cuda",
            dtype=torch.float32,
        )

        compare(x, groups, f"random seed={seed}")


def extreme_tests():
    shape = (2, 16, 64, 64)
    groups = 4

    factories = [
        ("zeros", lambda: torch.zeros(shape, device="cuda")),
        ("positive", lambda: torch.full(shape, 3.25, device="cuda")),
        ("negative", lambda: torch.full(shape, -3.25, device="cuda")),
        ("large", lambda: torch.full(shape, 1.0e4, device="cuda")),
        (
            "near integer",
            lambda: torch.randint(
                -5, 6, shape, device="cuda", dtype=torch.int32
            ).float() + 1.0e-6,
        ),
    ]

    for name, factory in factories:
        compare(factory(), groups, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    random_tests(args.seeds)
    extreme_tests()

    print(f"[PASS] robustness test completed: {args.seeds} random seeds")


if __name__ == "__main__":
    main()