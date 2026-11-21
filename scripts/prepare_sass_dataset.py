"""Audit task archives and prepare grouped SASS-to-CUDA chat datasets.

No GPU or third-party packages are needed unless --tokenizer is supplied.
Grouping is a conservative structural heuristic, not semantic deduplication.
"""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import os
import re


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class StructuralReference(ast.NodeTransformer):
    """Group identical reference structure even when constants differ."""

    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Must not exist")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tokenizer", help="Local Hugging Face tokenizer directory")
    parser.add_argument("--max-tokens", type=int, help="Exclude longer complete chat samples")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tokens is not None and (not args.tokenizer or args.max_tokens <= 0):
        parser.error("--max-tokens requires --tokenizer and a positive value")
    if not args.results.is_dir():
        parser.error("--results must be an existing directory")
    tasks = sorted(p for p in args.results.glob("task_*") if p.is_dir())
    if not tasks:
        parser.error("No task_* directories found")
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                                   trust_remote_code=False)
        if not tokenizer.chat_template:
            parser.error("Tokenizer needs a chat template for exact chat length counting")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config_path = args.output / "audit_config.json"
    config = {"results": str(args.results.resolve()), "tokenizer": args.tokenizer,
              "max_tokens": args.max_tokens, "version": 2}
    if args.tokenizer:
        config["tokenizer_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(args.tokenizer).glob("*")) if p.is_file()}
    if args.resume and config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume configuration/tokenizer differs; use a new output directory")
    write_json(config_path, config)
    journal = args.output / "lengths.jsonl"
    cached = {}
    if args.resume and journal.exists():
        with journal.open("rb+") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                    cached[record["task_id"]] = record
                except (ValueError, KeyError):
                    handle.truncate(offset)
                    break
    audit, samples = [], []
    for position, task in enumerate(tasks, 1):
        started = time.monotonic()
        stamp = [(str(p.relative_to(task)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(task.rglob("*")) if p.is_file() and "torch_compile" not in p.parts]
        fingerprint = sha(json.dumps(stamp))
        previous = cached.get(task.name)
        print(f"[{position}/{len(tasks)}] {task.name} START", flush=True)
        if previous and previous.get("fingerprint") == fingerprint:
            audit.append(previous)
            if previous["eligible"]:
                samples.append({"row": previous})
            print(f"  cached tokens={previous.get('total_tokens', 'unknown')}", flush=True)
            continue
        try:
            row = {"task_id": task.name, "archive": str(task.resolve()), "issues": []}
            files = {
                "reference": "model.py", "model_new": "model_new.py",
                "binding": "kernels/custom_ops_binding.cpp",
                "cuda": "kernels/custom_ops.cu", "sass": "kernels/custom_ops.sass",
                "ptx": "kernels/custom_ops.ptx", "cubin": "kernels/custom_ops.cubin",
                "verification": "verification.log", "meta": "meta.txt",
            }
            contents = {}
            row["files"] = {}
            for key, relative in files.items():
                path = task / relative
                if not path.is_file() or path.stat().st_size == 0:
                    row["issues"].append(f"missing_or_empty:{key}")
                    continue
                raw = path.read_bytes()
                row["files"][key] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                if key != "cubin":
                    contents[key] = raw.decode("utf-8", errors="replace")
            if "verification" in contents and not re.search(
                r"\[PASS\]\s+verify success\b", contents["verification"]
            ):
                row["issues"].append("no_verification_success_marker")
            native = sorted(str(p.relative_to(task)) for p in (task / "kernels").rglob("*")
                            if p.is_file() and p.suffix in {".cu", ".cpp"})
            if native != ["kernels/custom_ops.cu", "kernels/custom_ops_binding.cpp"]:
                row["issues"].append("additional_or_nonstandard_native_sources")
            if "reference" in contents:
                try:
                    tree = ast.parse(contents["reference"])
                    row["reference_ast_hash"] = sha(ast.dump(tree, include_attributes=False))
                    row["reference_structure_hash"] = sha(ast.dump(
                        StructuralReference().visit(tree), include_attributes=False))
                except SyntaxError:
                    row["issues"].append("invalid_reference_python")
            row["architectures"] = sorted(set(re.findall(
                r"\bsm_\d+[a-z]?\b", contents.get("sass", "") + contents.get("ptx", ""))))
            if not row["architectures"]:
                row["issues"].append("unknown_architecture")
            if not row["issues"]:
                # Full binding is an explicitly allowed input, identical at training/evaluation.
                messages = [
                    {"role": "user", "content": (
                        "根据以下 SASS 和固定 C++ binding 恢复完整 kernels/custom_ops.cu。\n"
                        "实现 binding 声明的所有 launcher，保持接口不变；不要修改 binding。\n"
                        "CUDA 文件不得依赖 torch/ATen/c10/pybind。只输出 CUDA 源码。\n"
                        f"目标架构：{', '.join(row['architectures'])}\n\n"
                        f"固定 binding：\n{contents['binding']}\n\nSASS：\n{contents['sass']}"
                    )},
                    {"role": "assistant", "content": contents["cuda"]},
                ]
                row["input_chars"] = len(messages[0]["content"])
                row["output_chars"] = len(messages[1]["content"])
                if tokenizer is not None:
                    row["total_tokens"] = len(tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False))
                    row["prompt_tokens"] = len(tokenizer.apply_chat_template(
                        messages[:1], tokenize=True, add_generation_prompt=True))
                    if args.max_tokens and row["total_tokens"] > args.max_tokens:
                        row["issues"].append("over_token_limit")
                if not row["issues"]:
                    row["message_hash"] = sha(json.dumps(messages, ensure_ascii=False))
                    samples.append({"row": row})
            row["eligible"] = not row["issues"]
        except Exception as exc:
            row = {"task_id": task.name, "archive": str(task.resolve()),
                   "eligible": False, "issues": [f"processing_error:{type(exc).__name__}:{exc}"]}
        row["fingerprint"] = fingerprint
        audit.append(row)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"  tokens={row.get('total_tokens', 'unknown')} issues={row['issues']} seconds={time.monotonic()-started:.1f}", flush=True)

    if args.audit_only:
        lengths = sorted(r["total_tokens"] for r in audit if "total_tokens" in r)
        report = {"archives": len(audit), "eligible": sum(r["eligible"] for r in audit),
                  "token_lengths": ({"min": lengths[0], "median": lengths[len(lengths)//2],
                    "p95": lengths[min(len(lengths)-1, int(.95*len(lengths)))], "max": lengths[-1]}
                    if lengths else None),
                  "over_limits": {str(n): sum(x > n for x in lengths) for n in (8192,16384,32768)}}
        write_json(args.output / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    # Union by reference structure or identical CUDA/SASS, including transitive matches.
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, sample in enumerate(samples):
        row = sample["row"]
        keys = [("reference", row["reference_structure_hash"]),
                ("cuda", row["files"]["cuda"]["sha256"]),
                ("sass", row["files"]["sass"]["sha256"])]
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[find(index)].append(sample)
    group_list = list(groups.values())
    random.Random(args.seed).shuffle(group_list)
    group_list.sort(key=len, reverse=True)
    split_rows = {name: [] for name in ("train", "valid", "test")}
    unique_count = len({s["row"]["message_hash"] for s in samples})
    targets = {"train": .8 * unique_count, "valid": .1 * unique_count, "test": .1 * unique_count}
    duplicate_count = 0
    for group in group_list:
        split = max(targets, key=lambda name: targets[name] - len(split_rows[name]))
        group_id = sha("\n".join(sorted(s["row"]["task_id"] for s in group)))[:16]
        unique = set()
        for sample in group:
            row = sample["row"]
            row.update(group_id=group_id, split=split)
            fingerprint = row["message_hash"]
            row["duplicate_removed"] = fingerprint in unique
            if fingerprint in unique:
                duplicate_count += 1
                continue
            unique.add(fingerprint)
            split_rows[split].append({"id": row["task_id"], "group_id": group_id,
                                      "archive": row["archive"]})
    for split, rows in split_rows.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in rows:
                task = Path(item["archive"])
                binding = (task / "kernels/custom_ops_binding.cpp").read_text()
                sass = (task / "kernels/custom_ops.sass").read_text()
                cuda = (task / "kernels/custom_ops.cu").read_text()
                row = next(r for r in audit if r["task_id"] == item["id"])
                prompt = ("根据以下 SASS 和固定 C++ binding 恢复完整 kernels/custom_ops.cu。\n"
                    "实现 binding 声明的所有 launcher，保持接口不变；不要修改 binding。\n"
                    "CUDA 文件不得依赖 torch/ATen/c10/pybind。只输出 CUDA 源码。\n"
                    f"目标架构：{', '.join(row['architectures'])}\n\n"
                    f"固定 binding：\n{binding}\n\nSASS：\n{sass}")
                handle.write(json.dumps({"id": item["id"], "group_id": item["group_id"],
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": cuda}]}, ensure_ascii=False) + "\n")
    write_jsonl(args.output / "manifest.jsonl", audit)
    lengths = sorted(row["total_tokens"] for row in audit if "total_tokens" in row)
    summary = {
        "seed": args.seed, "archives": len(tasks), "eligible_before_exact_dedup": len(samples),
        "excluded": sum(not row["eligible"] for row in audit),
        "issues": dict(Counter(issue for row in audit for issue in row["issues"])),
        "exact_duplicates_removed": duplicate_count, "groups": len(groups),
        "largest_group": max((len(g) for g in groups.values()), default=0),
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "tokenizer": args.tokenizer, "max_tokens": args.max_tokens,
        "token_lengths": ({"min": lengths[0], "median": lengths[len(lengths)//2],
                           "p95": lengths[min(len(lengths)-1, int(.95*len(lengths)))],
                           "max": lengths[-1]} if lengths else None),
        "limitations": [
            "Success markers record historical task-level verification, not independent kernel tests.",
            "Hashes fingerprint current files; they do not prove source/binary build provenance.",
            "AST grouping ignores constants but is not semantic similarity detection; manually inspect groups.",
            "Full binding is allowed input; inspect it for embedded computation or answer leakage.",
            "Token lengths remain unknown without the selected model tokenizer. No truncation is performed.",
            "Freeze the chosen split before model selection; changing filters can change split assignments.",
        ],
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
