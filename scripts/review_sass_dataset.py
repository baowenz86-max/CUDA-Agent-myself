"""Read-only dataset review; write reports separately from frozen inputs."""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tokenize


def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()


def load_split(path):
    rows = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["id"] in rows:
                    raise ValueError(f"Duplicate task in {path}: {item['id']}")
                rows[item["id"]] = {"hash": digest(item["messages"]), "group": item["group_id"]}
    return rows


def shingles(source):
    ast.parse(source)
    tokens = [t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
              if t.type not in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}]
    return {tuple(tokens[i:i+5]) for i in range(max(0, len(tokens)-4))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    splits = {s: load_split(args.data / f"{s}.jsonl") for s in ("train", "valid", "test")}
    structural = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        structural[f"{a}/{b}"] = {
            "task_overlap": sorted(splits[a].keys() & splits[b].keys()),
            "group_overlap": sorted({x["group"] for x in splits[a].values()} &
                                    {x["group"] for x in splits[b].values()})}
    holdouts = {}
    for s in ("valid", "test"):
        previous = load_split(args.previous / f"{s}.jsonl")
        holdouts[s] = previous.keys() == splits[s].keys() and all(
            previous[k]["hash"] == splits[s][k]["hash"] for k in previous)
    references, binding_reviews, errors = {}, [], []
    for split, items in splits.items():
        print(f"Reading {split}: {len(items)} tasks", flush=True)
        for task_id in items:
            archive = args.results / task_id
            try:
                references[task_id] = shingles((archive / "model.py").read_text())
                binding = (archive / "kernels/custom_ops_binding.cpp").read_text()
                markers = [m for m in ("at::", "torch::", "cublas", "cudnn", "for (", "for(", "while(", "while (")
                           if m in binding]
                binding_reviews.append({"id": task_id, "split": split,
                    "path": str(archive / "kernels/custom_ops_binding.cpp"),
                    "lines": len(binding.splitlines()), "review_markers": markers})
            except (OSError, SyntaxError, tokenize.TokenError) as exc:
                errors.append({"id": task_id, "error": str(exc)})
    pairs = []
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        print(f"Comparing {a}/{b} references", flush=True)
        for left in splits[a]:
            x = references.get(left, set())
            if not x:
                continue
            for right in splits[b]:
                y = references.get(right, set())
                if not y or min(len(x), len(y)) / max(len(x), len(y)) < .8:
                    continue
                intersection = len(x & y)
                score = intersection / (len(x) + len(y) - intersection)
                if score >= .8:
                    pairs.append({"left": left, "right": right, "splits": [a, b],
                                  "token_shingle_jaccard": round(score, 6)})
    pairs.sort(key=lambda x: x["token_shingle_jaccard"], reverse=True)
    binding_reviews.sort(key=lambda x: (len(x["review_markers"]), x["lines"]), reverse=True)
    report = {"counts": {s: len(r) for s, r in splits.items()},
              "holdout_contents_unchanged": holdouts, "split_overlap_checks": structural,
              "near_duplicate_pairs": len(pairs), "read_errors": errors,
              "note": "Similarity and binding markers are review hints, not proof of leakage or its absence."}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, obj in (("summary", report), ("near_duplicates", pairs), ("binding_review", binding_reviews)):
        (args.output / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
