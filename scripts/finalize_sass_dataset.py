"""Export a reviewed copy, excluding explicit training IDs only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exclude-train", nargs="+", required=True)
    args = p.parse_args()
    excluded = set(args.exclude_train)
    ids = {}
    for split in ("train", "valid", "test"):
        with (args.candidate / f"{split}.jsonl").open() as f:
            ids[split] = {json.loads(line)["id"] for line in f if line.strip()}
    if not excluded <= ids["train"]:
        raise ValueError(f"Exclusions not in training set: {sorted(excluded-ids['train'])}")
    if any(excluded & ids[s] for s in ("valid", "test")):
        raise ValueError("Cannot exclude held-out tasks")
    previous_train = set()
    with (args.previous / "train.jsonl").open() as f:
        previous_train = {json.loads(line)["id"] for line in f if line.strip()}
    # Compare IDs and content, independently of regenerated group labels/order.
    for split in ("valid", "test"):
        def hashes(path):
            with path.open() as f:
                return {r["id"]: hashlib.sha256(json.dumps(r["messages"], ensure_ascii=False).encode()).hexdigest()
                        for r in map(json.loads, (line for line in f if line.strip()))}
        if hashes(args.previous / f"{split}.jsonl") != hashes(args.candidate / f"{split}.jsonl"):
            raise ValueError(f"Frozen {split} changed")
    args.output.mkdir(parents=True, exist_ok=False)
    counts, fingerprints = {}, {}
    for split in ("train", "valid", "test"):
        count = 0
        with (args.candidate / f"{split}.jsonl").open() as source, (args.output / f"{split}.jsonl").open("w") as target:
            for line in source:
                if not line.strip():
                    continue
                if split == "train" and json.loads(line)["id"] in excluded:
                    continue
                target.write(line if line.endswith("\n") else line + "\n")
                count += 1
        counts[split] = count
        fingerprints[f"{split}.jsonl"] = hashlib.sha256((args.output / f"{split}.jsonl").read_bytes()).hexdigest()
    with (args.candidate / "manifest.jsonl").open() as source, (args.output / "manifest.jsonl").open("w") as target:
        for line in source:
            row = json.loads(line)
            if row["task_id"] in excluded:
                row.update(split="quarantine", quarantine_reason="manual_conservative_near_duplicate_exclusion")
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"candidate": str(args.candidate.resolve()), "previous": str(args.previous.resolve()),
              "splits": counts, "excluded_train_ids": sorted(excluded),
              "excluded_ids_in_previous_training": sorted(excluded & previous_train),
              "holdout_contents_unchanged": True, "sha256": fingerprints,
              "protocol": "Full fixed host binding supplied; recover custom CUDA, not host library computations.",
              "limitations": ["Similarity screening is heuristic, not exhaustive semantic deduplication.",
                              "Binding information contribution requires a no-SASS control.",
                              "Prior model results are not retroactively made leakage-free by these exclusions."]}
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
