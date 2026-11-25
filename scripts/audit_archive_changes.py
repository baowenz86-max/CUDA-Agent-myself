"""Read-only comparison of current archives with a previous dataset manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--previous", type=Path,
                        default=Path("data/sass_cuda_compact_32k_v1"))
    args = parser.parse_args()
    if not args.results.is_dir():
        parser.error("Results directory does not exist")
    old = {}
    with (args.previous / "manifest.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                old[row["task_id"]] = row
    current = {p.name: p for p in args.results.glob("task_*") if p.is_dir()}
    paths = {"reference": "model.py", "model_new": "model_new.py",
             "binding": "kernels/custom_ops_binding.cpp", "cuda": "kernels/custom_ops.cu",
             "sass": "kernels/custom_ops.sass", "ptx": "kernels/custom_ops.ptx",
             "cubin": "kernels/custom_ops.cubin"}
    common = sorted(old.keys() & current.keys())
    print(f"旧归档={len(old)} 当前归档={len(current)} 待比较={len(common)}", flush=True)
    changed, details = {}, Counter()
    for index, task_id in enumerate(common, 1):
        differences = []
        for key, relative in paths.items():
            previous = old[task_id].get("files", {}).get(key)
            if not previous:
                continue
            path = current[task_id] / relative
            if not path.is_file():
                differences.append(key + ":missing")
            elif file_hash(path) != previous["sha256"]:
                differences.append(key + ":changed")
        if differences:
            changed[task_id] = differences
            details.update(differences)
        print(f"[{index}/{len(common)}] {task_id}: {differences or '未变'}", flush=True)
    print("新增任务ID数：", len(current.keys() - old.keys()))
    print("旧任务已不存在：", len(old.keys() - current.keys()))
    print("旧任务已记录文件发生变化：", len(changed))
    print("变化类型：", dict(details))
    for split in ("train", "valid", "test"):
        with (args.previous / f"{split}.jsonl").open() as handle:
            ids = {json.loads(line)["id"] for line in handle if line.strip()}
        affected = sorted(ids & changed.keys())
        missing = sorted(ids - current.keys())
        print(f"旧 {split}: 变化={len(affected)}, 缺失={len(missing)}")
        print("变化示例：", affected[:10])
        print("缺失示例：", missing[:10])
    print("变化详情示例：")
    for task_id in list(changed)[:10]:
        print(task_id, changed[task_id])


if __name__ == "__main__":
    main()
