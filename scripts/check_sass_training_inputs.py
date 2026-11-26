"""Check frozen dataset fingerprints and local weight-file presence without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path


def check(data, model):
    summary = json.loads((data / "summary.json").read_text())
    for split in ("train", "valid", "test"):
        name = f"{split}.jsonl"
        path = data / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != summary["sha256"][name]:
            raise ValueError(f"Dataset changed: {path}")
        with path.open() as handle:
            count = sum(bool(line.strip()) for line in handle)
        if count != summary["splits"][split] or not count:
            raise ValueError(f"Invalid sample count: {path}")
        print(f"{split}: {count}; SHA256 verified", flush=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        json.loads((model / name).read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise ValueError("Empty weight index")
    for name in shards:
        path = model / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty weight: {path}")
    print(f"Weight shards: {len(shards)}; file presence verified (not content integrity)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    check(args.data, args.model)


if __name__ == "__main__":
    main()
