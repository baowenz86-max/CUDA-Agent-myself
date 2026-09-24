"""Download the CUDA-Agent training dataset into a local cache."""

import argparse
from pathlib import Path

from cuda_agent.workflow import DATASET_NAME, REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Hugging Face dataset cache directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required; install project dependencies first."
        ) from exc

    dataset = load_dataset(DATASET_NAME, cache_dir=str(args.cache_dir))
    print(dataset)
    print(dataset["train"][0])


if __name__ == "__main__":
    main()
