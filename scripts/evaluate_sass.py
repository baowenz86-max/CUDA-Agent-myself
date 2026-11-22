"""Rebuild original and recovered CUDA in isolated directories; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import tempfile


def run(command, directory, env, timeout, name):
    started = time.monotonic()
    with (directory / f"{name}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"returncode": None, "timeout": True, "seconds": time.monotonic()-started}
    return {"returncode": code, "timeout": False, "seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for each run")
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    python = str(args.python.absolute())
    if not Path(python).is_file():
        parser.error("Verification Python does not exist")
    config = json.loads((args.predictions / "generation_config.json").read_text())
    # Freeze inputs so completed results are never reused with different source.
    digest = hashlib.sha256()
    paths = [args.predictions / "generation_config.json", Path(__file__)]
    paths += sorted((repo / "agent_workdir/utils").glob("*.py"))
    paths += [repo / "agent_workdir" / n for n in ("binding.cpp", "binding_registry.h")]
    for task_id in config["ids"]:
        if not task_id.startswith("task_") or not task_id[5:].isdigit():
            raise ValueError(f"Invalid task ID {task_id}")
        paths += [args.results / task_id / n for n in
                  ("model.py", "model_new.py", "kernels/custom_ops.cu", "kernels/custom_ops_binding.cpp")]
        paths.append(args.predictions / task_id / "result.json")
        candidate = args.predictions / task_id / "custom_ops.cu"
        if candidate.exists():
            paths.append(candidate)
    for path in paths:
        digest.update(str(path.absolute()).encode())
        digest.update(path.read_bytes())
    settings = {"input_hash": digest.hexdigest(), "python": python, "timeout": args.timeout,
                "env": {k: os.environ.get(k) for k in
                        ("CUDA_HOME", "CC", "CXX", "CUDAHOSTCXX", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS")}}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    settings_path = args.output / "evaluation_config.json"
    if args.resume and any(args.output.iterdir()) and not settings_path.exists():
        raise ValueError("Cannot resume old output without evaluation_config.json; use a new directory")
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Evaluation inputs/settings changed; use a new output directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    rows = []
    for position, task_id in enumerate(config["ids"], 1):
        if not task_id.startswith("task_") or not task_id[5:].isdigit():
            raise ValueError(f"Invalid task ID {task_id}")
        saved = args.output / task_id / "result.json"
        if args.resume and saved.exists():
            try:
                cached = json.loads(saved.read_text())
                if cached.get("id") == task_id and "status" in cached:
                    rows.append(cached)
                    print(f"[{position}/{len(config['ids'])}] {task_id} cached: {cached['status']}", flush=True)
                    continue
            except ValueError:
                pass
        task_root = args.output / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        # Preserve interrupted attempts; never reuse their possibly stale binary.
        attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=task_root)).absolute()
        print(f"[{position}/{len(config['ids'])}] {task_id} original replay", flush=True)
        archive = args.results / task_id
        seed = 2026 + int(task_id[5:]) * 100
        row = {"id": task_id, "seed": seed, "attempt": str(attempt)}

        def prepare(kind, cuda):
            directory = attempt / kind
            directory.mkdir(parents=True)
            shutil.copytree(repo / "agent_workdir/utils", directory / "utils",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in ("binding.cpp", "binding_registry.h"):
                shutil.copy2(repo / "agent_workdir" / name, directory / name)
            for name in ("model.py", "model_new.py"):
                shutil.copy2(archive / name, directory / name)
            (directory / "kernels").mkdir()
            shutil.copy2(archive / "kernels/custom_ops_binding.cpp",
                         directory / "kernels/custom_ops_binding.cpp")
            shutil.copy2(cuda, directory / "kernels/custom_ops.cu")
            return directory

        def evaluate(directory):
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["CUDA_AGENT_RUN_DIR"] = str(directory / ".correctness")
            env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
            result = {"compile": run([python, "-m", "utils.compile"], directory,
                                      env, args.timeout, "compile")}
            if result["compile"]["returncode"] != 0:
                result["status"] = "compile_failed"
                return result
            result["verification"] = run([python, "-m", "utils.verification", "--seed", str(seed)],
                                         directory, env, args.timeout, "verification")
            path = directory / ".correctness/verification.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            result["verification_report"] = report
            passed = result["verification"]["returncode"] == 0 and report.get("status") == "passed"
            result["status"] = "passed" if passed else "verification_failed"
            return result

        try:
            original = prepare("original", archive / "kernels/custom_ops.cu")
            row["original"] = evaluate(original)
            if row["original"]["status"] != "passed":
                row["status"] = "original_replay_failed"
            else:
                prediction = args.predictions / task_id
                generation = json.loads((prediction / "result.json").read_text())
                row["generation"] = generation
                if generation.get("truncated") or not generation.get("format_ok"):
                    row["status"] = "generation_failed"
                else:
                    print(f"  {task_id} recovered compile/verify", flush=True)
                    candidate = prepare("recovered", prediction / "custom_ops.cu")
                    (candidate / ".correctness").mkdir()
                    shutil.copy2(original / ".correctness/reference_fixture.pt",
                                 candidate / ".correctness/reference_fixture.pt")
                    row["recovered"] = evaluate(candidate)
                    row["status"] = row["recovered"]["status"]
        except Exception as exc:
            row.update(status="infrastructure_error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        (args.output / task_id).mkdir(parents=True, exist_ok=True)
        temporary = saved.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(saved)
        print(f"  {task_id}: {row['status']}", flush=True)
    evaluable = [r for r in rows if r.get("original", {}).get("status") == "passed"
                 and r["status"] != "infrastructure_error"]
    summary = {"tasks": len(rows), "evaluable": len(evaluable),
               "infrastructure_errors": sum(r["status"] == "infrastructure_error" for r in rows),
               "original_replay_failed": sum(r["status"] == "original_replay_failed" for r in rows),
               "compiled": sum(r.get("recovered", {}).get("compile", {}).get("returncode") == 0 for r in evaluable),
               "functionally_correct": sum(r["status"] == "passed" for r in evaluable),
               "status_by_task": {r["id"]: r["status"] for r in rows}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
